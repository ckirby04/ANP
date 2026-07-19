"""Deterministic data splits for BraTS-MEN.

Splits are not generated here. They are read from the nnU-Net
`splits_final.json` that already exists for this dataset, so every arm and seed
sees byte-identical case lists and the split is reproducible from config alone.
A digest of the resolved split is exposed so a run can record which split it
trained on.

Single fold by design: train on fold 0, select on its validation split, report
headline numbers on the held-out test cohort. No ensembling, no multi-fold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# The 150-case test cohort is not preprocessed. Only the 850 training cases
# have .b2nd files; test cases live as raw NIfTI and must be preprocessed at
# evaluation time.
TEST_LABEL_SUFFIX = ".nii.gz"


@dataclass(frozen=True)
class Split:
    """Resolved case identifiers for one fold."""

    fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]

    def __post_init__(self):
        overlap = set(self.train) & set(self.val)
        if overlap:
            raise ValueError(
                f"fold {self.fold}: {len(overlap)} identifiers in both train and "
                f"val, e.g. {sorted(overlap)[:3]}")

    @property
    def digest(self) -> str:
        """Stable hash of the split, for recording alongside run results."""
        payload = json.dumps(
            {"fold": self.fold, "train": sorted(self.train), "val": sorted(self.val)},
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.train) + len(self.val)


def load_split(preprocessed_dir: str | Path, fold: int = 0) -> Split:
    """Read fold `fold` from the dataset's splits_final.json."""
    path = Path(preprocessed_dir) / "splits_final.json"
    if not path.is_file():
        raise FileNotFoundError(f"no splits file at {path}")

    with open(path) as fh:
        folds = json.load(fh)
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold {fold} out of range; file has {len(folds)} folds")

    entry = folds[fold]
    # Sorted so the ordering is independent of how the file was written.
    return Split(fold=fold,
                 train=tuple(sorted(entry["train"])),
                 val=tuple(sorted(entry["val"])))


def available_identifiers(preprocessed_data_dir: str | Path) -> tuple[str, ...]:
    """Case identifiers that actually have preprocessed .b2nd data on disk."""
    d = Path(preprocessed_data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"no preprocessed data folder at {d}")
    return tuple(sorted(
        p.name[:-5] for p in d.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")))


def verify_split_available(split: Split,
                           preprocessed_data_dir: str | Path) -> None:
    """Fail loudly if the split names cases that were never preprocessed.

    Silently dropping missing cases would change the effective training set
    between arms without changing any config, which is exactly the kind of
    difference that invalidates a controlled comparison.
    """
    have = set(available_identifiers(preprocessed_data_dir))
    missing = sorted((set(split.train) | set(split.val)) - have)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} case(s) in fold {split.fold} have no preprocessed "
            f"data in {preprocessed_data_dir}, e.g. {missing[:3]}")


def heldout_identifiers(raw_dataset_dir: str | Path) -> tuple[str, ...]:
    """Held-out test cohort identifiers, read from labelsTs.

    These have no preprocessed .b2nd data and must be preprocessed at
    evaluation time.
    """
    d = Path(raw_dataset_dir) / "labelsTs"
    if not d.is_dir():
        raise FileNotFoundError(f"no labelsTs folder at {d}")
    return tuple(sorted(
        p.name[: -len(TEST_LABEL_SUFFIX)] for p in d.glob(f"*{TEST_LABEL_SUFFIX}")))

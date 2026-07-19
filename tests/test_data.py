"""Tests for splits and patch sampling against the real dataset on disk.

These deliberately touch the real data. A loader that passes on synthetic
arrays and fails on the actual preprocessed cases is worth nothing, and the
spec forbids generating substitute data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import SEG_OOB, BraTSMENPatches  # noqa: E402
from data.splits import (  # noqa: E402
    available_identifiers,
    heldout_identifiers,
    load_split,
    verify_split_available,
)

NNUNET = Path(r"G:\BraTS-MEN\nnUNet")
PREPROCESSED = NNUNET / "nnUNet_preprocessed" / "Dataset002_BraTS_MEN"
DATA_DIR = PREPROCESSED / "nnUNetPlans_3d_fullres"
RAW = NNUNET / "nnUNet_raw" / "Dataset002_BraTS_MEN"

PATCH = (128, 160, 112)
N_CHANNELS = 4

pytestmark = pytest.mark.skipif(
    not DATA_DIR.is_dir(), reason=f"preprocessed data not present at {DATA_DIR}")


# --- splits ---------------------------------------------------------------

def test_fold0_sizes_and_disjointness():
    split = load_split(PREPROCESSED, fold=0)
    assert len(split.train) == 680
    assert len(split.val) == 170
    assert len(split) == 850
    assert not set(split.train) & set(split.val)


def test_split_is_deterministic():
    a, b = load_split(PREPROCESSED, 0), load_split(PREPROCESSED, 0)
    assert a.train == b.train and a.val == b.val
    assert a.digest == b.digest


def test_folds_differ():
    assert load_split(PREPROCESSED, 0).digest != load_split(PREPROCESSED, 1).digest


def test_every_split_case_is_preprocessed():
    verify_split_available(load_split(PREPROCESSED, 0), DATA_DIR)


def test_preprocessed_cohort_size():
    assert len(available_identifiers(DATA_DIR)) == 850


def test_heldout_test_cohort():
    ids = heldout_identifiers(RAW)
    assert len(ids) == 150
    # The test cohort must not leak into the training split.
    assert not set(ids) & set(load_split(PREPROCESSED, 0).train)


def test_missing_case_raises():
    split = load_split(PREPROCESSED, 0)
    bad = type(split)(fold=0, train=split.train + ("BraTS_MEN_99999",), val=split.val)
    with pytest.raises(FileNotFoundError, match="no preprocessed data"):
        verify_split_available(bad, DATA_DIR)


# --- one real case --------------------------------------------------------

def test_load_one_case_shapes_and_labels():
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    identifier = load_split(PREPROCESSED, 0).train[0]
    ds = nnUNetDatasetBlosc2(str(DATA_DIR), [identifier])
    data, seg, seg_prev, properties = ds.load_case(identifier)

    assert data.ndim == 4 and data.shape[0] == N_CHANNELS
    assert seg.ndim == 4 and seg.shape[0] == 1
    assert data.shape[1:] == seg.shape[1:]
    assert data.dtype == np.float32
    assert seg_prev is None

    # -1 is present on disk: nnU-Net writes it outside the skull-stripped brain.
    labels = np.unique(np.asarray(seg[:]))
    assert set(labels.tolist()) <= {-1, 0, 1}, f"unexpected labels {labels}"

    assert "class_locations" in properties
    assert "spacing" in properties


def test_no_ignore_label_in_dataset():
    """Guards the precondition that makes the -1 remap match the reference run.

    nnUNetTrainer maps -1 to background via RemoveLabelTansform(-1, 0) and only
    masks voxels out of the loss when has_ignore_label is True. If an ignore
    label were ever added to dataset.json, collapsing -1 to background would
    start training on voxels the reference run excluded, changing the effective
    class balance and invalidating the baseline comparison.
    """
    import json

    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    with open(PREPROCESSED / "dataset.json") as fh:
        labels = json.load(fh)["labels"]
    assert labels == {"background": 0, "lesion": 1}

    lm = LabelManager(labels, regions_class_order=None)
    assert not lm.has_ignore_label
    assert lm.ignore_label is None
    assert not lm.has_regions


def test_disk_negative_one_is_outside_brain_and_lesion_free():
    """The invariant that licenses collapsing -1 to background.

    If -1 ever marked an unannotated but in-brain region, or contained lesion,
    the remap in BraTSMENPatches would be silently mislabelling voxels as
    background and inflating every Dice number in the study.
    """
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    ids = load_split(PREPROCESSED, 0).train[:3]
    ds = nnUNetDatasetBlosc2(str(DATA_DIR), list(ids))
    for identifier in ids:
        data, seg, _, _ = ds.load_case(identifier)
        s = np.asarray(seg[:])[0]
        outside = s == SEG_OOB
        assert outside.any(), f"{identifier}: expected an outside-brain region"
        # -1 coincides exactly with all modalities being zero.
        np.testing.assert_array_equal(
            outside, (np.asarray(data[:]) == 0).all(axis=0),
            err_msg=f"{identifier}: -1 region is not the zero-intensity region")
        assert not (s[outside] == 1).any(), f"{identifier}: lesion inside -1 region"


# --- patch sampling -------------------------------------------------------

def _dataset(seed=0, length=8, oversample=0.33, n_cases=8):
    ids = load_split(PREPROCESSED, 0).train[:n_cases]
    return BraTSMENPatches(DATA_DIR, ids, PATCH, length=length, seed=seed,
                           oversample_foreground=oversample)


def test_patch_shape_dtype_and_labels():
    sample = _dataset()[0]
    assert sample["data"].shape == (N_CHANNELS, *PATCH)
    assert sample["seg"].shape == (1, *PATCH)
    assert sample["data"].dtype == np.float32
    assert sample["seg"].dtype == np.int8
    # -1 padding must be resolved; this dataset has no ignore label.
    assert set(np.unique(sample["seg"]).tolist()) <= {0, 1}
    assert np.isfinite(sample["data"]).all()


def test_sampling_is_reproducible():
    a, b = _dataset(seed=7)[3], _dataset(seed=7)[3]
    assert a["identifier"] == b["identifier"]
    np.testing.assert_array_equal(a["data"], b["data"])
    np.testing.assert_array_equal(a["seg"], b["seg"])


def test_seed_changes_sampling():
    a, b = _dataset(seed=1)[0], _dataset(seed=2)[0]
    differs = (a["identifier"] != b["identifier"]
               or not np.array_equal(a["data"], b["data"]))
    assert differs, "different seeds produced an identical patch"


def test_epoch_changes_sampling():
    d = _dataset(seed=0)
    a = d[0]
    d.set_epoch(1)
    b = d[0]
    differs = (a["identifier"] != b["identifier"]
               or not np.array_equal(a["data"], b["data"]))
    assert differs, "different epochs produced an identical patch"


def test_full_oversampling_hits_foreground():
    """With oversample=1.0 every patch should contain foreground.

    This is the test that catches an off-by-one in the class_locations channel
    index, which would otherwise degrade silently into random sampling.
    """
    d = _dataset(seed=3, length=6, oversample=1.0)
    for i in range(6):
        seg = d[i]["seg"]
        assert (seg > 0).any(), f"patch {i} has no foreground under oversample=1.0"


def test_no_oversampling_is_mostly_background():
    d = _dataset(seed=5, length=12, oversample=0.0)
    fg = sum((d[i]["seg"] > 0).any() for i in range(12))
    # Tumors are large in this cohort, so random patches often clip one. The
    # assertion is only that foreground is not being forced.
    assert fg < 12, "oversample=0.0 still produced foreground in every patch"

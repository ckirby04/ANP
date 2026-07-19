"""Patch sampling over the preprocessed BraTS-MEN cases.

Reads the blosc2 `.b2nd` volumes nnU-Net already produced for this dataset
rather than re-preprocessing. Blosc2 arrays are lazy: slicing decompresses only
the intersecting blocks, and the chunk and block sizes were tuned at
preprocessing time for patch-sized reads. Cropping therefore never materializes
a full 4 x 240 x 240 x 155 volume, which is what keeps the loop compute-bound
rather than data-bound.

Foreground oversampling uses the `class_locations` sampled into each case's
`.pkl` at preprocessing time, matching nnU-Net's own dataloader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from torch.utils.data import Dataset

# -1 reaches the seg patch from two independent places:
#
#   1. On disk. nnU-Net's preprocessor writes -1 wherever the nonzero mask is
#      False, i.e. outside the skull-stripped brain. Verified on this cohort:
#      the -1 region agrees with "all four modalities zero" to 100 percent, is
#      about 56 percent of each cropped volume, and never contains lesion.
#   2. As padding, when a sampled patch overhangs the volume.
#
# This dataset has no ignore label, so neither source is annotation-bearing and
# both are non-lesion. They are remapped to background before returning.
SEG_OOB = -1


class BraTSMENPatches(Dataset):
    """Random patches from preprocessed cases, with foreground oversampling.

    One __getitem__ yields one patch, not one case. `length` therefore sets the
    number of samples per epoch independently of cohort size, so the schedule
    is identical across arms regardless of how many cases a split holds.

    Sampling is deterministic given `seed`: sample i of epoch e always draws the
    same case and the same patch location.
    """

    def __init__(self,
                 preprocessed_data_dir: str | Path,
                 identifiers: tuple[str, ...],
                 patch_size: tuple[int, int, int],
                 length: int,
                 seed: int,
                 oversample_foreground: float = 0.33):
        if not 0.0 <= oversample_foreground <= 1.0:
            raise ValueError(
                f"oversample_foreground must be in [0, 1], got {oversample_foreground}")
        if not identifiers:
            raise ValueError("no identifiers given")

        self.folder = str(preprocessed_data_dir)
        self.identifiers = tuple(identifiers)
        self.patch_size = tuple(patch_size)
        self.length = int(length)
        self.seed = int(seed)
        self.oversample_foreground = float(oversample_foreground)
        self._epoch = 0

        # Constructed lazily per worker process: blosc2 handles do not survive
        # being pickled across a DataLoader worker boundary.
        self._ds: nnUNetDatasetBlosc2 | None = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the sampling stream so epochs differ but stay reproducible."""
        self._epoch = int(epoch)

    def _dataset(self) -> nnUNetDatasetBlosc2:
        if self._ds is None:
            self._ds = nnUNetDatasetBlosc2(self.folder, list(self.identifiers))
        return self._ds

    def _rng(self, index: int) -> np.random.Generator:
        # Seeded per (seed, epoch, index) so a sample is reproducible without
        # any dependence on worker count or iteration order.
        return np.random.default_rng((self.seed, self._epoch, index))

    def _bbox(self, rng, shape: tuple[int, ...],
              properties: dict) -> list[list[int]]:
        """Half-open [start, end) bbox over the trailing spatial dims.

        Bounds are deliberately allowed to run outside the volume; crop_and_pad_nd
        pads the overhang. Constraining them inward instead would bias sampling
        against the volume edges.
        """
        spatial = shape[-3:]
        want_fg = rng.random() < self.oversample_foreground

        centre = None
        if want_fg:
            locs = properties.get("class_locations") or {}
            # Real foreground classes only; keys are int labels or region tuples.
            usable = [k for k, v in locs.items() if v is not None and len(v) > 0]
            if usable:
                key = usable[rng.integers(len(usable))]
                voxels = locs[key]
                # Each entry is (channel, x, y, z); drop the leading channel index.
                centre = np.asarray(voxels[rng.integers(len(voxels))])[1:]

        bbox = []
        for dim, patch in zip(range(3), self.patch_size):
            extent = spatial[dim]
            if centre is not None:
                start = int(centre[dim]) - patch // 2
                # Keep the foreground voxel inside the patch while allowing the
                # patch itself to overhang the volume.
                lo, hi = min(0, extent - patch), max(0, extent - patch)
                start = int(np.clip(start, lo, hi))
            else:
                lo, hi = min(0, extent - patch), max(0, extent - patch)
                start = int(rng.integers(lo, hi + 1))
            bbox.append([start, start + patch])
        return bbox

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        ds = self._dataset()

        identifier = self.identifiers[rng.integers(len(self.identifiers))]
        data, seg, _seg_prev, properties = ds.load_case(identifier)

        bbox = self._bbox(rng, data.shape, properties)
        # Lazy blosc2 slice: only the intersecting blocks are decompressed.
        data_patch = crop_and_pad_nd(data, bbox, 0)
        seg_patch = crop_and_pad_nd(seg, bbox, SEG_OOB)

        seg_patch = np.asarray(seg_patch)
        # Collapses both the on-disk outside-brain region and the out-of-bounds
        # padding to background. See the SEG_OOB note above.
        seg_patch[seg_patch == SEG_OOB] = 0

        return {
            "data": np.ascontiguousarray(data_patch, dtype=np.float32),
            "seg": np.ascontiguousarray(seg_patch, dtype=np.int8),
            "identifier": identifier,
        }

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
import torch
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
#
# This matches the reference fold-0 run rather than diverging from it.
# nnUNetTrainer inserts RemoveLabelTansform(-1, 0) into both its training and
# validation transform pipelines (nnUNetTrainer.py:800 and :855), and its
# ignore-label masking path (:1056-1060) is gated on has_ignore_label, which is
# False for this dataset since dataset.json declares only {background, lesion}.
# The reference run therefore also trained on these voxels as background, so
# the 178 s/epoch and ~0.90 pseudo-Dice reference points describe the same
# class balance the arms here face. test_no_ignore_label_in_dataset guards the
# precondition.
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
                 oversample_foreground: float = 0.33,
                 transforms=None,
                 start_index: int = 0):
        """
        Args:
            patch_size: the size actually cropped from the volume. For training
                this is nnU-Net's *initial* patch size, which is larger than the
                final one so that rotation and scaling in SpatialTransform have
                margin to work with; the transform pipeline crops down to the
                final size. Sampling at the final size instead would introduce
                rotation border artifacts the reference run does not have.
            transforms: a batchgeneratorsv2 transform, applied per sample with
                keyword args `image` and `segmentation`. When given, the
                segmentation is passed through with -1 intact, because
                MaskImageTransform reads seg < 0 to mask the image before
                RemoveLabelTansform(-1, 0) strips it. Remapping -1 earlier would
                silently turn that transform into a no-op.
        """
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
        self.transforms = transforms
        # Global sample offset. The training loader runs as one continuous
        # stream for the whole run rather than one stream per epoch, because
        # spawning DataLoader workers costs about 21 s on Windows and paying
        # that every epoch would dominate the schedule. Resuming sets this to
        # the already-consumed sample count so the stream picks up where it
        # left off instead of replaying.
        self.start_index = int(start_index)
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
        # Seeded per (seed, epoch, global index) so a sample is reproducible
        # without any dependence on worker count or iteration order.
        return np.random.default_rng(
            (self.seed, self._epoch, self.start_index + index))

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

        if self.transforms is None:
            # Raw mode, used by tests and diagnostics. Nothing downstream will
            # consume the -1, so collapse both the on-disk outside-brain region
            # and the out-of-bounds padding to background here.
            seg_patch[seg_patch == SEG_OOB] = 0
            return {
                "data": np.ascontiguousarray(data_patch, dtype=np.float32),
                "seg": np.ascontiguousarray(seg_patch, dtype=np.int8),
                "identifier": identifier,
            }

        # Transform mode. -1 is left intact: MaskImageTransform masks the image
        # where seg < 0, and RemoveLabelTansform(-1, 0) later in the same
        # pipeline does the remap. Matches nnU-Net's own dataloader, which
        # hands float32 image and int16 segmentation per sample with no batch
        # dimension.
        out = self.transforms(
            image=torch.from_numpy(np.ascontiguousarray(data_patch)).float(),
            segmentation=torch.from_numpy(np.ascontiguousarray(seg_patch)).to(torch.int16),
        )
        return {
            "data": out["image"],
            "target": out["segmentation"],
            "identifier": identifier,
        }

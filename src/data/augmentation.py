"""Augmentation pipeline, reproduced from nnU-Net's own trainer.

The four arms need identical augmentation, which any consistent choice would
satisfy. The stronger requirement is that the 178 s/epoch and 0.90 pseudo-Dice
reference points from the prior fold-0 run only transfer if the pipeline
matches that run. So these call nnU-Net's own static transform builders rather
than reimplementing them.

Two properties of that pipeline are load-bearing and easy to get wrong:

  - Training crops an *initial* patch size larger than the final one, and
    SpatialTransform crops down after rotating and scaling. For patch
    [128, 160, 112] the initial size is [224, 228, 189]. Sampling at the final
    size would leave rotation border artifacts.
  - MaskImageTransform reads seg < 0 to mask the image, and only afterwards
    does RemoveLabelTansform(-1, 0) strip the -1. The segmentation must
    therefore reach the pipeline with -1 intact.
"""

from __future__ import annotations

import numpy as np
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

ROTATION_HALF_RANGE_DEG = 30.0
SCALE_RANGE = (0.7, 1.4)
# nnU-Net derives the initial patch size using this narrower range, not SCALE_RANGE.
INITIAL_PATCH_SCALE_RANGE = (0.85, 1.25)


def configure_augmentation(patch_size: tuple[int, ...]) -> dict:
    """Rotation range, dummy-2D flag, initial patch size and mirror axes.

    Mirrors nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size
    for the 3D case, without needing a trainer instance.
    """
    patch = np.array(patch_size)
    do_dummy_2d = (max(patch) / patch[0]) > ANISO_THRESHOLD
    if do_dummy_2d:
        raise NotImplementedError(
            f"patch {tuple(patch_size)} is anisotropic enough to trigger nnU-Net's "
            "dummy 2D augmentation path, which this project does not reproduce")

    rot = (-ROTATION_HALF_RANGE_DEG / 360 * 2 * np.pi,
           ROTATION_HALF_RANGE_DEG / 360 * 2 * np.pi)
    initial = get_patch_size(patch[-3:], rot, rot, rot, INITIAL_PATCH_SCALE_RANGE)

    return {
        "rotation_for_DA": rot,
        "do_dummy_2d_data_aug": False,
        "initial_patch_size": tuple(int(i) for i in initial),
        "mirror_axes": (0, 1, 2),
    }


def build_train_transforms(patch_size: tuple[int, ...],
                           deep_supervision_scales: list | None,
                           use_mask_for_norm: list[bool]):
    """nnU-Net's training transform stack for this configuration."""
    cfg = configure_augmentation(patch_size)
    return nnUNetTrainer.get_training_transforms(
        patch_size=patch_size,
        rotation_for_DA=cfg["rotation_for_DA"],
        deep_supervision_scales=deep_supervision_scales,
        mirror_axes=cfg["mirror_axes"],
        do_dummy_2d_data_aug=cfg["do_dummy_2d_data_aug"],
        use_mask_for_norm=use_mask_for_norm,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    )


def build_val_transforms(deep_supervision_scales: list | None):
    """Validation transforms: label cleanup and deep-supervision downsampling only."""
    return nnUNetTrainer.get_validation_transforms(
        deep_supervision_scales=deep_supervision_scales,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    )

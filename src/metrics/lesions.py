"""Lesion-wise stratification by size.

This is where silent bugs live, so the conventions are stated rather than
implied:

  - Components are labelled on the GROUND TRUTH, never on the prediction. The
    buckets describe the lesions that exist, not the ones the model found.
  - Size is equivalent spherical diameter in mm, derived from the component's
    volume and the voxel spacing.
  - A ground-truth lesion counts as detected if the prediction overlaps it by
    at least `min_overlap` of its voxels. This is a per-lesion criterion, not
    a per-voxel one: detecting one voxel of a large lesion is not a detection.
  - Components below `min_voxels` are dropped before anything else.

On this dataset the floor matters. 92.5 percent of sub-5mm components in
BraTS-MEN are dural-tail fragmentation, rim partial-volume or label speckle,
and 64 percent are a single voxel. The floor is configurable so the same
analysis can be reported floored and unfloored, and neither is hardcoded as
"the" answer. See docs/dataset_report.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Upper bound of each bucket in mm of equivalent diameter; the last is open.
DEFAULT_BUCKETS = ((0.0, 5.0, "<5mm"), (5.0, 10.0, "5-10mm"), (10.0, np.inf, ">10mm"))
DEFAULT_MIN_VOXELS = 10
DEFAULT_MIN_OVERLAP = 0.1


@dataclass(frozen=True)
class Lesion:
    label: int
    n_voxels: int
    volume_mm3: float
    equiv_diam_mm: float
    bucket: str
    detected: bool
    overlap_voxels: int

    @property
    def overlap_fraction(self) -> float:
        return self.overlap_voxels / self.n_voxels if self.n_voxels else 0.0


def equivalent_diameter_mm(n_voxels: int, voxel_volume_mm3: float) -> float:
    """Diameter of the sphere with the same volume as the component."""
    volume = n_voxels * voxel_volume_mm3
    return 2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


def assign_bucket(diameter_mm: float, buckets=DEFAULT_BUCKETS) -> str:
    for lo, hi, name in buckets:
        # Half-open [lo, hi) so a lesion exactly at a boundary lands in the
        # upper bucket and cannot be counted twice.
        if lo <= diameter_mm < hi:
            return name
    return buckets[-1][2]


def label_lesions(gt: np.ndarray,
                  spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  min_voxels: int = DEFAULT_MIN_VOXELS,
                  connectivity: int = 1):
    """Connected components of the ground truth, above the size floor.

    Returns the label image and the retained component labels. `connectivity`
    is 1 for face-adjacency (6-neighbour in 3D) and 3 for full 26-neighbour
    adjacency; face-adjacency is the conservative default, since 26-neighbour
    merges lesions that touch only at a corner.
    """
    g = np.asarray(gt).astype(bool)
    structure = ndimage.generate_binary_structure(g.ndim, connectivity)
    lab, n = ndimage.label(g, structure=structure)
    if n == 0:
        return lab, []

    counts = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if counts[i] >= min_voxels]
    return lab, keep


def lesion_table(gt: np.ndarray,
                 pred: np.ndarray,
                 spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 min_voxels: int = DEFAULT_MIN_VOXELS,
                 min_overlap: float = DEFAULT_MIN_OVERLAP,
                 buckets=DEFAULT_BUCKETS,
                 connectivity: int = 1) -> list[Lesion]:
    """One record per ground-truth lesion, with its bucket and detection flag."""
    g = np.asarray(gt).astype(bool)
    p = np.asarray(pred).astype(bool)
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch: gt {g.shape} vs pred {p.shape}")

    voxel_volume = float(np.prod(spacing))
    lab, keep = label_lesions(g, spacing, min_voxels, connectivity)

    out = []
    for label in keep:
        mask = lab == label
        n_vox = int(mask.sum())
        overlap = int(np.logical_and(mask, p).sum())
        diam = equivalent_diameter_mm(n_vox, voxel_volume)
        out.append(Lesion(
            label=int(label),
            n_voxels=n_vox,
            volume_mm3=n_vox * voxel_volume,
            equiv_diam_mm=diam,
            bucket=assign_bucket(diam, buckets),
            detected=(overlap / n_vox) >= min_overlap if n_vox else False,
            overlap_voxels=overlap,
        ))
    return out


def stratified_summary(lesions: list[Lesion], buckets=DEFAULT_BUCKETS) -> dict:
    """Per-bucket lesion counts and detection sensitivity.

    Sensitivity is per-lesion: detected lesions over lesions present. A bucket
    with no lesions reports sensitivity None rather than 0.0 or NaN, so that
    aggregating across subjects can skip it instead of averaging in a number
    that means "no data".
    """
    summary = {}
    for _, _, name in buckets:
        in_bucket = [x for x in lesions if x.bucket == name]
        n = len(in_bucket)
        n_det = sum(1 for x in in_bucket if x.detected)
        summary[name] = {
            "n_lesions": n,
            "n_detected": n_det,
            "sensitivity": (n_det / n) if n else None,
        }
    return summary


def dice_by_bucket(gt: np.ndarray,
                   pred: np.ndarray,
                   spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                   min_voxels: int = DEFAULT_MIN_VOXELS,
                   buckets=DEFAULT_BUCKETS,
                   connectivity: int = 1) -> dict:
    """Dice computed over the union of ground-truth lesions in each bucket.

    Only the ground-truth side is restricted to the bucket. The prediction is
    restricted to the same region, so this measures how well the model
    segments lesions of that size, not whether it produced false positives
    elsewhere. Whole-tumour Dice remains the headline number; this is a
    breakdown of it.
    """
    from .dice import dice

    g = np.asarray(gt).astype(bool)
    p = np.asarray(pred).astype(bool)
    lab, keep = label_lesions(g, spacing, min_voxels, connectivity)
    voxel_volume = float(np.prod(spacing))

    out = {}
    for _, _, name in buckets:
        region = np.zeros_like(g)
        for label in keep:
            mask = lab == label
            diam = equivalent_diameter_mm(int(mask.sum()), voxel_volume)
            if assign_bucket(diam, buckets) == name:
                region |= mask
        if not region.any():
            out[name] = None
            continue
        out[name] = dice(np.logical_and(p, region), region)
    return out

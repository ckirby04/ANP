"""Metrics tests on synthetic masks with hand-computable answers.

The lesion-wise stratification is where silent bugs live, so the cases here
are deliberately adversarial: empty predictions, empty ground truth, disjoint
masks, and components sitting exactly on the size floor and exactly on a
bucket boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics.dice import confusion, dice  # noqa: E402
from metrics.lesions import (  # noqa: E402
    DEFAULT_BUCKETS,
    assign_bucket,
    dice_by_bucket,
    equivalent_diameter_mm,
    label_lesions,
    lesion_table,
    stratified_summary,
)

SHAPE = (40, 40, 40)


def blank():
    return np.zeros(SHAPE, dtype=bool)


def cube(arr, corner, size, value=True):
    i, j, k = corner
    arr[i:i + size, j:j + size, k:k + size] = value
    return arr


def sphere_voxels(arr, centre, n_voxels):
    """Place approximately n_voxels as a compact blob at centre."""
    ci, cj, ck = centre
    r = int(np.ceil((3 * n_voxels / (4 * np.pi)) ** (1 / 3))) + 2
    ii, jj, kk = np.mgrid[ci - r:ci + r + 1, cj - r:cj + r + 1, ck - r:ck + r + 1]
    d = (ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2
    order = np.argsort(d.ravel())[:n_voxels]
    coords = np.array(np.unravel_index(order, d.shape))
    arr[coords[0] + ci - r, coords[1] + cj - r, coords[2] + ck - r] = True
    return arr


# --- Dice ------------------------------------------------------------------

def test_dice_identical_is_one():
    a = cube(blank(), (10, 10, 10), 6)
    assert dice(a, a) == 1.0


def test_dice_empty_empty_is_one_not_nan_or_zero():
    """A correct negative is perfect agreement, not a failure or a NaN."""
    e = blank()
    result = dice(e, e)
    assert result == 1.0
    assert not np.isnan(result)


def test_dice_empty_prediction_is_zero():
    gt = cube(blank(), (10, 10, 10), 6)
    assert dice(blank(), gt) == 0.0


def test_dice_empty_ground_truth_is_zero():
    pred = cube(blank(), (10, 10, 10), 6)
    assert dice(pred, blank()) == 0.0


def test_dice_disjoint_is_zero():
    a = cube(blank(), (2, 2, 2), 5)
    b = cube(blank(), (30, 30, 30), 5)
    assert np.logical_and(a, b).sum() == 0
    assert dice(a, b) == 0.0


def test_dice_half_overlap_is_two_thirds():
    # |A| = |B| = 8, |A and B| = 4  ->  2*4 / (8+8) = 0.5
    a = blank()
    a[0, 0, 0:8] = True
    b = blank()
    b[0, 0, 4:12] = True
    assert dice(a, b) == pytest.approx(0.5)


def test_dice_is_symmetric():
    a = cube(blank(), (5, 5, 5), 7)
    b = cube(blank(), (8, 8, 8), 7)
    assert dice(a, b) == pytest.approx(dice(b, a))


def test_dice_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        dice(np.zeros((4, 4, 4)), np.zeros((4, 4, 5)))


def test_confusion_counts():
    gt = blank()
    gt[0, 0, 0:10] = True
    pred = blank()
    pred[0, 0, 5:15] = True
    c = confusion(pred, gt)
    assert c["tp"] == 5
    assert c["fp"] == 5
    assert c["fn"] == 5
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == int(np.prod(SHAPE))


# --- geometry --------------------------------------------------------------

def test_equivalent_diameter_of_known_volume():
    # A sphere of diameter 10mm has volume 523.6 mm3.
    assert equivalent_diameter_mm(523.6, 1.0) == pytest.approx(10.0, rel=1e-3)


def test_equivalent_diameter_scales_with_spacing():
    # Same voxel count at 2mm isotropic is 8x the volume, so 2x the diameter.
    d1 = equivalent_diameter_mm(1000, 1.0)
    d2 = equivalent_diameter_mm(1000, 8.0)
    assert d2 == pytest.approx(2 * d1)


def test_bucket_boundaries_are_half_open():
    """A lesion exactly at a boundary lands in the upper bucket, exactly once."""
    assert assign_bucket(4.999) == "<5mm"
    assert assign_bucket(5.0) == "5-10mm"
    assert assign_bucket(9.999) == "5-10mm"
    assert assign_bucket(10.0) == ">10mm"
    assert assign_bucket(1000.0) == ">10mm"


def test_every_diameter_lands_in_exactly_one_bucket():
    for d in np.linspace(0.0, 50.0, 501):
        hits = [name for lo, hi, name in DEFAULT_BUCKETS if lo <= d < hi]
        assert len(hits) == 1, f"diameter {d} matched {hits}"


# --- connected components and the size floor -------------------------------

def test_floor_drops_components_below_threshold():
    gt = blank()
    sphere_voxels(gt, (10, 10, 10), 200)   # large, kept
    gt[30, 30, 30] = True                  # single voxel, dropped
    _, keep = label_lesions(gt, min_voxels=10)
    assert len(keep) == 1


def test_component_exactly_at_floor_is_kept():
    """Exactly 10 voxels at a floor of 10 is kept: the floor is inclusive."""
    gt = blank()
    gt[5, 5, 0:10] = True
    assert int(gt.sum()) == 10
    _, keep = label_lesions(gt, min_voxels=10)
    assert len(keep) == 1


def test_component_one_below_floor_is_dropped():
    gt = blank()
    gt[5, 5, 0:9] = True
    assert int(gt.sum()) == 9
    _, keep = label_lesions(gt, min_voxels=10)
    assert len(keep) == 0


def test_unfloored_keeps_single_voxels():
    """The sensitivity analysis path must retain what the primary path drops."""
    gt = blank()
    gt[30, 30, 30] = True
    _, keep = label_lesions(gt, min_voxels=1)
    assert len(keep) == 1


def test_face_adjacency_does_not_merge_corner_touching_lesions():
    gt = blank()
    cube(gt, (10, 10, 10), 3)
    cube(gt, (13, 13, 13), 3)   # touches the first only at a corner
    _, keep_face = label_lesions(gt, min_voxels=1, connectivity=1)
    _, keep_full = label_lesions(gt, min_voxels=1, connectivity=3)
    assert len(keep_face) == 2
    assert len(keep_full) == 1


# --- lesion-wise detection -------------------------------------------------

def test_detection_is_per_lesion_not_per_voxel():
    """One overlapping voxel in a large lesion is not a detection."""
    gt = blank()
    sphere_voxels(gt, (20, 20, 20), 500)
    pred = blank()
    coord = np.argwhere(gt)[0]
    pred[tuple(coord)] = True

    lesions = lesion_table(gt, pred, min_overlap=0.1)
    assert len(lesions) == 1
    assert lesions[0].overlap_voxels == 1
    assert not lesions[0].detected


def test_detection_at_overlap_threshold():
    gt = blank()
    gt[5, 5, 0:100 if SHAPE[2] > 100 else 40] = True
    n = int(gt.sum())
    pred = blank()
    pred[5, 5, 0:int(np.ceil(0.1 * n))] = True
    lesions = lesion_table(gt, pred, min_overlap=0.1)
    assert lesions[0].detected


def test_lesions_are_labelled_on_ground_truth_not_prediction():
    """Extra predicted blobs must not create lesion records."""
    gt = blank()
    sphere_voxels(gt, (10, 10, 10), 300)
    pred = gt.copy()
    sphere_voxels(pred, (30, 30, 30), 300)   # false positive elsewhere

    lesions = lesion_table(gt, pred)
    assert len(lesions) == 1, "a predicted blob was counted as a ground-truth lesion"
    assert lesions[0].detected


def test_empty_ground_truth_yields_no_lesions():
    pred = cube(blank(), (10, 10, 10), 5)
    assert lesion_table(blank(), pred) == []


def test_empty_prediction_detects_nothing():
    gt = blank()
    sphere_voxels(gt, (10, 10, 10), 300)
    lesions = lesion_table(gt, blank())
    assert len(lesions) == 1
    assert not lesions[0].detected
    assert lesions[0].overlap_voxels == 0


def test_disjoint_prediction_detects_nothing():
    gt = sphere_voxels(blank(), (10, 10, 10), 300)
    pred = sphere_voxels(blank(), (30, 30, 30), 300)
    lesions = lesion_table(gt, pred)
    assert len(lesions) == 1
    assert not lesions[0].detected


# --- stratified summary ----------------------------------------------------

def test_stratification_assigns_known_sizes_to_known_buckets():
    """Three lesions built to fall in three different buckets."""
    gt = blank()
    # equivalent diameters: ~2.0mm (4 vox), ~7.3mm (200 vox), ~12.4mm (1000 vox)
    sphere_voxels(gt, (5, 5, 5), 4)
    sphere_voxels(gt, (20, 5, 5), 200)
    sphere_voxels(gt, (30, 30, 30), 1000)

    lesions = lesion_table(gt, gt, min_voxels=1)
    by_bucket = {x.bucket for x in lesions}
    assert by_bucket == {"<5mm", "5-10mm", ">10mm"}

    summary = stratified_summary(lesions)
    for name in ("<5mm", "5-10mm", ">10mm"):
        assert summary[name]["n_lesions"] == 1
        assert summary[name]["sensitivity"] == 1.0


def test_empty_bucket_reports_none_not_zero():
    """None means no data; 0.0 would poison a mean across subjects."""
    gt = sphere_voxels(blank(), (20, 20, 20), 1000)
    summary = stratified_summary(lesion_table(gt, gt))
    assert summary[">10mm"]["sensitivity"] == 1.0
    assert summary["<5mm"]["n_lesions"] == 0
    assert summary["<5mm"]["sensitivity"] is None


def test_floor_changes_the_small_bucket_only():
    """The primary and sensitivity analyses must differ where expected."""
    gt = blank()
    sphere_voxels(gt, (20, 20, 20), 1000)
    for c in [(2, 2, 2), (2, 2, 8), (2, 8, 2)]:
        gt[c] = True   # single-voxel speckle

    floored = stratified_summary(lesion_table(gt, gt, min_voxels=10))
    unfloored = stratified_summary(lesion_table(gt, gt, min_voxels=1))
    assert floored["<5mm"]["n_lesions"] == 0
    assert unfloored["<5mm"]["n_lesions"] == 3
    assert floored[">10mm"]["n_lesions"] == unfloored[">10mm"]["n_lesions"] == 1


# --- dice by bucket --------------------------------------------------------

def test_dice_by_bucket_perfect_prediction():
    gt = blank()
    sphere_voxels(gt, (10, 10, 10), 200)
    sphere_voxels(gt, (30, 30, 30), 1000)
    out = dice_by_bucket(gt, gt)
    assert out["5-10mm"] == pytest.approx(1.0)
    assert out[">10mm"] == pytest.approx(1.0)
    assert out["<5mm"] is None


def test_dice_by_bucket_isolates_the_missed_lesion():
    """Missing the small lesion must not drag down the large-lesion bucket."""
    gt = blank()
    sphere_voxels(gt, (10, 10, 10), 200)      # 5-10mm
    big = sphere_voxels(blank(), (30, 30, 30), 1000)
    gt |= big

    pred = big.copy()                          # large found, small missed
    out = dice_by_bucket(gt, pred)
    assert out["5-10mm"] == pytest.approx(0.0)
    assert out[">10mm"] == pytest.approx(1.0)


def test_dice_by_bucket_ignores_false_positives_outside_lesions():
    # 1000 voxels is 12.4mm equivalent diameter, so this is the >10mm bucket.
    # 500 voxels would be 9.85mm and land in 5-10mm instead.
    gt = sphere_voxels(blank(), (10, 10, 10), 1000)
    pred = gt.copy()
    sphere_voxels(pred, (32, 32, 32), 500)     # false positive far away
    out = dice_by_bucket(gt, pred)
    assert out[">10mm"] == pytest.approx(1.0)
    # Whole-tumour Dice must still see the false positive.
    assert dice(pred, gt) < 1.0


def test_anisotropic_spacing_changes_bucket_assignment():
    """A component's bucket depends on spacing, not voxel count alone."""
    gt = blank()
    sphere_voxels(gt, (20, 20, 20), 200)
    iso = lesion_table(gt, gt, spacing=(1.0, 1.0, 1.0))
    coarse = lesion_table(gt, gt, spacing=(2.0, 2.0, 2.0))
    assert iso[0].bucket == "5-10mm"
    assert coarse[0].bucket == ">10mm"
    assert coarse[0].volume_mm3 == pytest.approx(8 * iso[0].volume_mm3)

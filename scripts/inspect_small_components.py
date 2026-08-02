"""Characterize sub-5mm connected components in the BraTS-MEN ground truth.

Motivating question: BraTS-MEN is meningioma, which is typically solitary,
dural-based and large. A preliminary scan found ~34 percent of connected
components have equivalent diameter < 5mm. That is not what meningioma looks
like, so those components are more plausibly dural-tail fragmentation,
partial-volume edges at the tumor rim, or label speckle than genuine small
lesions.

The discriminator is distance to the dominant lesion. A component that is a
fragment of the main tumor sits within a few voxels of it. A genuine
independent small lesion sits far away in separate parenchyma.

Outputs (written to results/diagnostics/, gitignored):
  small_component_stats.csv  one row per connected component
  small_component_montage.png  20 sampled sub-5mm components over T1c
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DATA_ROOT_HINT, default_raw_dir  # noqa: E402

RAW = Path(default_raw_dir()) if default_raw_dir() else None
SMALL_DIAM_MM = 5.0
# Equivalent-diameter threshold expressed as a voxel count, at 1mm isotropic.
SMALL_MAX_VOXELS = (4.0 / 3.0) * np.pi * (SMALL_DIAM_MM / 2.0) ** 3


def equiv_diameter(n_voxels):
    """Diameter of the sphere with the same volume, in mm at 1mm isotropic."""
    return 2.0 * (3.0 * n_voxels / (4.0 * np.pi)) ** (1.0 / 3.0)


def component_records(case_id, seg):
    """Label seg and describe every component relative to the largest one."""
    lab, n = ndimage.label(seg > 0)
    if n == 0:
        return []

    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    sizes = np.asarray(sizes, dtype=np.int64)
    dominant = int(np.argmax(sizes)) + 1

    # Distance, in voxels, from every position to the nearest dominant-lesion
    # voxel. Sampling this at a component's voxels gives its separation from
    # the main tumor, which is what separates a fragment from a real lesion.
    dist_to_dominant = ndimage.distance_transform_edt(lab != dominant)

    centroids = ndimage.center_of_mass(seg > 0, lab, index=range(1, n + 1))

    rows = []
    for i in range(1, n + 1):
        mask = lab == i
        n_vox = int(sizes[i - 1])
        sep = 0.0 if i == dominant else float(dist_to_dominant[mask].min())
        rows.append(
            {
                "case_id": case_id,
                "component_id": i,
                "n_voxels": n_vox,
                "equiv_diam_mm": round(float(equiv_diameter(n_vox)), 3),
                "is_dominant": int(i == dominant),
                "n_components_in_case": n,
                "dominant_n_voxels": int(sizes[dominant - 1]),
                "sep_from_dominant_mm": round(sep, 3),
                "centroid_i": round(float(centroids[i - 1][0]), 1),
                "centroid_j": round(float(centroids[i - 1][1]), 1),
                "centroid_k": round(float(centroids[i - 1][2]), 1),
            }
        )
    return rows


def summarize(rows):
    small = [r for r in rows if r["equiv_diam_mm"] < SMALL_DIAM_MM and not r["is_dominant"]]
    total = len(rows)
    print(f"\ncomponents total: {total}")
    print(f"sub-{SMALL_DIAM_MM:.0f}mm non-dominant components: {len(small)}")
    if not small:
        return

    vox = np.array([r["n_voxels"] for r in small])
    sep = np.array([r["sep_from_dominant_mm"] for r in small])

    print("\nvoxel count of sub-5mm components:")
    for lo, hi, name in [(1, 1, "1 voxel"), (2, 2, "2 voxels"),
                         (3, 5, "3-5"), (6, 10, "6-10"), (11, 66, "11-65")]:
        c = int(((vox >= lo) & (vox <= hi)).sum())
        print(f"  {name:>8}: {c:4d}  ({100.0 * c / len(small):5.1f}%)")

    print("\nseparation from the dominant lesion:")
    for lo, hi, name in [(0, 2, "<=2mm  (touching / rim)"),
                         (2, 5, "2-5mm  (dural tail range)"),
                         (5, 10, "5-10mm"),
                         (10, 1e9, ">10mm  (independent)")]:
        c = int(((sep > lo) & (sep <= hi)).sum()) if lo else int((sep <= hi).sum())
        print(f"  {name:<26}: {c:4d}  ({100.0 * c / len(small):5.1f}%)")

    frag = int(((sep <= 5.0) | (vox <= 2)).sum())
    print(f"\nattributable to fragmentation or speckle "
          f"(<=5mm from dominant, or <=2 voxels): {frag}/{len(small)} "
          f"({100.0 * frag / len(small):.1f}%)")

    solitary = [r for r in small if r["dominant_n_voxels"] < SMALL_MAX_VOXELS]
    print(f"cases whose dominant lesion is itself sub-5mm: {len(solitary)}")


def render(rows, out_png, raw, n_show=20, seed=0):
    """Montage of sampled sub-5mm components over their T1c source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    small = [r for r in rows if r["equiv_diam_mm"] < SMALL_DIAM_MM and not r["is_dominant"]]
    if not small:
        print("nothing to render")
        return

    rng = random.Random(seed)
    # Spread the sample across the separation range so the montage is not all
    # rim fragments or all distant components.
    small_sorted = sorted(small, key=lambda r: r["sep_from_dominant_mm"])
    idx = np.linspace(0, len(small_sorted) - 1, min(n_show, len(small_sorted)))
    picks = [small_sorted[int(i)] for i in idx]
    rng.shuffle(picks)

    cols, half = 5, 24
    figrows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(figrows, cols, figsize=(3.0 * cols, 3.2 * figrows))
    axes = np.atleast_1d(axes).ravel()

    cache = {}
    for ax, r in zip(axes, picks):
        cid = r["case_id"]
        if cid not in cache:
            t1c = np.asanyarray(
                nib.load(str(raw / "imagesTr" / f"{cid}_0001.nii.gz")).dataobj)
            seg = np.asanyarray(
                nib.load(str(raw / "labelsTr" / f"{cid}.nii.gz")).dataobj)
            cache[cid] = (t1c, seg)
        t1c, seg = cache[cid]

        ci, cj, ck = int(r["centroid_i"]), int(r["centroid_j"]), int(r["centroid_k"])
        i0, i1 = max(0, ci - half), min(t1c.shape[0], ci + half)
        j0, j1 = max(0, cj - half), min(t1c.shape[1], cj + half)

        img = t1c[i0:i1, j0:j1, ck]
        msk = seg[i0:i1, j0:j1, ck]

        lo, hi = np.percentile(img[img > 0], [1, 99]) if (img > 0).any() else (0, 1)
        ax.imshow(img.T, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        if (msk > 0).any():
            ax.contour(msk.T, levels=[0.5], colors="red", linewidths=0.9)
        ax.plot(ci - i0, cj - j0, "+", color="cyan", markersize=9, markeredgewidth=1.2)
        ax.set_title(
            f"{cid.replace('BraTS_MEN_', '')}  {r['n_voxels']}vox "
            f"{r['equiv_diam_mm']:.1f}mm\nsep {r['sep_from_dominant_mm']:.1f}mm",
            fontsize=8)
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle("Sub-5mm ground-truth components over T1c "
                 "(red = GT contour, cyan + = component centroid)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"\nwrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=250,
                    help="number of training cases to scan (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/diagnostics"))
    ap.add_argument("--raw-dir", type=Path, default=RAW,
                    help="nnU-Net raw dataset dir; defaults to ANP_DATA_ROOT")
    args = ap.parse_args()

    raw = args.raw_dir
    if raw is None:
        raise SystemExit("no raw dataset directory.\n\n" + DATA_ROOT_HINT)

    labels = sorted((raw / "labelsTr").glob("*.nii.gz"))
    if not labels:
        raise SystemExit(f"no labels found under {raw / 'labelsTr'}")
    if args.n_cases:
        random.Random(args.seed).shuffle(labels)
        labels = labels[: args.n_cases]

    rows = []
    for n, path in enumerate(labels, 1):
        case_id = path.name.replace(".nii.gz", "")
        seg = np.asanyarray(nib.load(str(path)).dataobj)
        rows.extend(component_records(case_id, seg))
        if n % 50 == 0:
            print(f"  scanned {n}/{len(labels)} cases")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "small_component_stats.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  ({len(rows)} components from {len(labels)} cases)")

    summarize(rows)
    render(rows, args.out_dir / "small_component_montage.png", raw,
           seed=args.seed)


if __name__ == "__main__":
    main()

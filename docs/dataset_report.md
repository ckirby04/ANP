# Dataset report: BraTS-MEN

Build step 1. Findings recorded before any data loader was written.

## Location and layout

`G:\BraTS-MEN` is not a raw dataset directory. It is a completed prior project
(JANNUS v1.4 transfer experiment, hand-off dated 2026-05-04) with the imaging
data inside it. Two usable views of the same 1000 cases:

**`extracted\BraTS-MEN-Train\`** — 1000 case directories named
`BraTS-MEN-XXXXX-000`, each with `-t1n`, `-t1c`, `-t2f`, `-t2w`, `-seg` as
`.nii.gz`. Segmentations carry original BraTS labels `{1,2,3}`; individual
cases often lack some labels (observed `{0,2,3}` and `{0,3}`).

**`nnUNet\nnUNet_raw\Dataset002_BraTS_MEN\`** — the same data converted to
nnU-Net layout, and the view this project uses:

| | count |
|---|---|
| `imagesTr` | 3400 files (850 cases x 4 channels) |
| `labelsTr` | 850 |
| `imagesTs` | 600 files (150 cases x 4 channels) |
| `labelsTs` | 150 |

Channel order is `0=T1, 1=T1_Gd, 2=FLAIR, 3=T2`. Labels are already binarized
to `{background: 0, lesion: 1}`, i.e. whole tumor.

## Geometry

All volumes are 240 x 240 x 155 at 1.0 x 1.0 x 1.0 mm isotropic, uniform
across the cohort. This is co-registered, skull-stripped BraTS 2023 output, so
there is no meaningful spacing distribution to characterize.

## Reused preprocessing

nnU-Net preprocessing is already complete for `3d_fullres`: 850 cases,
15.2 GB, blosc2 `.b2nd` format. `splits_final.json` defines 5 folds; fold 0 is
680 train / 170 val.

The auto-configured plan, which is the object this experiment studies:

| | |
|---|---|
| patch size | `[128, 160, 112]` |
| batch size | 2 |
| stages | 6 |
| features per stage | `[32, 64, 128, 256, 320, 320]` |
| strides | `[1,1,1], [2,2,2] x4, [2,2,1]` |
| convs per stage | 2 encoder, 2 decoder |
| norm / nonlin | InstanceNorm3d / LeakyReLU |
| architecture | `dynamic_network_architectures.PlainConvUNet` |

A prior fold-0 `3d_fullres` run of 315 epochs exists with logs, giving a hard
timing anchor of **178 s/epoch** on the 16 GB card. Pseudo-Dice reached
~0.90-0.95 and was flat from roughly epoch 250.

## Lesion size stratification: the sub-5mm bucket is an artifact

A first scan over 120 cases found 34.4 percent of connected components had
equivalent diameter below 5mm. That is not consistent with meningioma, which
is typically solitary, dural-based and large, so those components were
characterized directly before any stratified metric was built.

`scripts\inspect_small_components.py` labels every ground-truth component and
measures its separation from the dominant lesion in the same case. A fragment
of the main tumor sits within a few voxels of it; a genuine independent lesion
sits far away in separate parenchyma.

Over 250 cases, 418 components, 120 of them sub-5mm and non-dominant:

**Size** — 64.2 percent are a *single voxel*. 74.2 percent are two voxels or
fewer. Only 8.3 percent exceed 10 voxels.

**Separation from the dominant lesion**

| | share |
|---|---|
| <= 2mm (touching / rim) | 60.0% |
| 2-5mm (dural tail range) | 24.2% |
| 5-10mm | 2.5% |
| > 10mm (independent) | 13.3% |

**92.5 percent** are attributable to fragmentation or speckle (within 5mm of
the dominant lesion, or two voxels or fewer). No case in the sample had a
dominant lesion that was itself sub-5mm.

Visual inspection of 20 sampled components over their T1c source
(`results\diagnostics\small_component_montage.png`, gitignored) agrees: the
components are single voxels hugging the contour of a large tumor, rim
partial-volume, and dural-tail fragmentation.

### Consequence

The sub-5mm bucket measures **agreement with annotation noise**, not
small-lesion detection. Specifically:

- It does **not** support any claim about micrometastasis or small-lesion
  detection. That claim requires a metastases dataset (BraTS-METS) and is out
  of scope here.
- The stratified metric is still computed and reported, honestly labeled as
  what it is. A 10-voxel minimum component size is the primary configuration,
  with the unfloored numbers reported as a sensitivity analysis.
- The primary readout of this experiment, the layer-wise sparsity trajectory,
  is unaffected. It stands on its own regardless of what the sub-5mm bucket
  turns out to be.

# Measured phenotype data

Elemental composition (ICP-MS) and reflectance-spectroscopy indices for the
same *Sorghum bicolor* and *Populus trichocarpa* plants the transcriptomes came
from. **Measured data — not produced by this repository or by any model in it.**

`E1.0_Sorghum_Poplar_ICP-MS_Spec_total.txt`, tab-separated, 354 rows x 58
columns, one row per sample.

## Columns

| | |
|---|---|
| 1–5 | `Sample ID`, `Tissue`, `Treatment`, `Timepoint`, `Species` |
| 6–25 | ICP-MS elements: `B11 Na23 Mg26 Al27 P31 S34 K39 Ca44 Fe54 Mn55 Co59 Ni60 Cu63 Zn66 As75 Se78 Rb85 Sr88 Mo98 Cd111` (isotope-suffixed) |
| 26 | unnamed, empty — a spacer in the source spreadsheet, read by pandas as `Unnamed: 25` |
| 27–56 | reflectance indices: `NDVI800 NDVI850 reNDVI PRI1-3 WBI NDWI1-2 SR1-5 DWSI SIPI ARI1-2 CRI1-2 WCRI CCI CCI2 NIRv R2131 R710 VIS_mean NIR_mean SWIR1_mean SWIR2_mean` |
| 57–58 | pigments: `Chl`, `Car` |

## Design

Two species x two tissues x five treatments x seven timepoints
(`0 h`, `1 h`, `2 d`, `4 d`, `7 d`, `14 d`, `21 d`), unbalanced:

| Species | Tissue | Control | FeEX | FeLim | ZnEx | ZnLim |
|---|---|---|---|---|---|---|
| *P. trichocarpa* | Leaf | 19 | 15 | 15 | 15 | 15 |
| *P. trichocarpa* | Root | 22 | 18 | 18 | 18 | 18 |
| *S. bicolor* | Leaf | 21 | 15 | 15 | 17 | 15 |
| *S. bicolor* | Root | 24 | 18 | 18 | 20 | 18 |

## What the manuscript uses

Only the **Leaf** rows, and only **Control** and **FeLim**. The two insets in
`fig_photo_etc` panel B carry leaf iron (`Fe54`) and total chlorophyll (`Chl`),
plotted as the FeLim-minus-Control difference. The remaining treatments,
the root rows, and the other 50-odd columns are not used by any figure; they
are included because they are the same measurement campaign and splitting them
would make the file harder to interpret, not easier.

The spectral indices were also used off-figure, to rank which measured
phenotypes track the electron-transport-chain trajectory most closely
(water-status indices came out on top); that ranking informed which insets were
chosen but does not itself appear in the manuscript.

## How the figures find it

`Paper_Figures/fig_photo_etc.py` reads this path by default. Override with
`BIOFLUX_PHENOTYPE_FILE` to point at a different copy.

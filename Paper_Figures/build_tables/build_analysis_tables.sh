#!/usr/bin/env bash
# Rebuild the analysis tables the manuscript figures read.
#
# Output goes to the directory io_utils.cross_species_dir() resolves to --
# Paper_Figures/analysis_tables/ by default, or BIOFLUX_CROSS_DIR.
#
# Four scripts, in this order. The order was derived by tracing every script's
# actual reads and writes of the table directory with a Python audit hook on
# `open`, not by reading the source: several reach the directory through
# io_utils rather than a literal path. The dependencies are:
#
#   analyze_both_species.py       -> curated_flux_both_species.tsv
#                                    network_resid_both_species.tsv
#   analyze_all_central_carbon.py -> all_central_carbon_{classification,timecourse}.tsv
#   find_limiting_genes.py        -> limiting_genes_ranking.tsv   <- curated_flux
#   glk_regulon_summary.py        -> glk_regulon_shared9.tsv      <- limiting_genes_ranking
#
# Those six tables are all the figures read. Verified by holding every other
# table out of the directory and confirming all nine figures still render
# byte-identical. In particular no iron_*.tsv table is read by any figure;
# analyze_both_species.py emits three as byproducts and they are removed below.
#
# Needs both external data dependencies -- see ../README.md and
# ../../requirements-runtime.txt:
#   BIOFLUX_PLANTSEED_DIR   ModelSEED/PlantSEED @ v2.5
#   BIOFLUX_RNASEQ_DIR      RNASeq_Enzyme_Abundance @ bioflux-preprint-260813
#
# Sorghum-only is enough for the Sorghum panels; the Poplar inputs are not
# distributed by either repository.
#
# About 20 seconds. Deterministic: two rebuilds produce byte-identical tables.
set -euo pipefail
cd "$(dirname "$0")"
RUN="${BIOFLUX_PYTHON:-micromamba run -n bf-runtime python}"

$RUN analyze_both_species.py
$RUN analyze_all_central_carbon.py
$RUN ../figures_src/find_limiting_genes.py   # vendored there; it imports
                                             # io_utils from its own directory
$RUN glk_regulon_summary.py

# byproducts no figure reads
OUT="$($RUN -c 'import sys,os; sys.path.insert(0,os.path.join("..","figures_src")); import io_utils; print(io_utils.cross_species_dir())')"
rm -f "$OUT"/iron_*.tsv "$OUT"/all_central_carbon_flux.tsv "$OUT"/glk_regulon_shared9.tex

echo "[done] $(ls "$OUT" | wc -l) tables in $OUT"

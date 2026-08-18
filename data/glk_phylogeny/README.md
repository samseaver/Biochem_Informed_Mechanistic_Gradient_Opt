# GLK paralog assignment — the phylogenetic basis

Why the manuscript calls `Sobic.010G096300` **SbGLK1** and `Sobic.003G002600`
**SbGLK2**, and why Poplar's two copies are called **PtGLK-A** and **PtGLK-B**
instead of GLK1 and GLK2.

This matters because the labels are load-bearing: the text reports that the ten
photosynthetic subunits favour SbGLK1 over SbGLK2 (mean Pearson *r* +0.86 vs
+0.45), and reads that separation as the sub-functionalized, bundle-sheath /
mesophyll-compartmentalized GLK pair of C4 grasses (Wang et al., 2012). If the
two Sorghum genes were labelled the other way round, that sentence would say
the opposite.

## The short version

The grass and eudicot GLK duplications are **independent**. Neither Arabidopsis
gene is specifically related to either grass clade — both share the same last
common ancestor with both — so there is no 1:1 orthology to be had, and an
Arabidopsis-derived label cannot name a grass paralog at all. Not because the
labels come out crossed, but because the mapping does not exist. The Sorghum
names therefore come from **which grass clade each gene falls in**, not from
which Arabidopsis gene they score best against.

- **SbGLK1** = `Sobic.010G096300` — sister to maize `Zm00001d044785`, in the
  grass clade that also holds rice `LOC_Os06g24070` and *Brachypodium*
  `Bradi1g43710`. This is grass GLK1.
- **SbGLK2** = `Sobic.003G002600` — sister to maize `Zm00001d039260`, in the
  clade with rice `LOC_Os01g13740` and `Bradi2g08287`.
- **PtGLK-A / PtGLK-B** = `Potri.007G136901` / `Potri.017G015800` — each other's
  immediate sister, a recent Salicaceae whole-genome-duplication pair. They are
  not separable into GLK1 and GLK2, hence the neutral A/B labels.

## The tree

Pruned from OrthoFinder orthogroup OG0003894 to the twelve informative tips:
the two Sorghum and two Poplar genes, the two Arabidopsis genes, and the maize
/ rice / *Brachypodium* copies that define the two grass clades.

```
             , Sbicolor Sobic.010G096300.1     SbGLK1
           __|
          |  |_ Zmays    Zm00001d044785_T001    
    ______|
   |      |__ Osativa  LOC_Os06g24070.1       
   |      |
   |      |_____ Bdistach Bradi1g43710.1         
  _|
 | |       _ Sbicolor Sobic.003G002600.1     SbGLK2
 | |     _|
 | |    | |___ Zmays    Zm00001d039260_T002    
 | |____|
 |      | ___ Bdistach Bradi2g08287.1         
_|      ||
 |       |___ Osativa  LOC_Os01g13740.1       
 |
 |     _______ Athalian AT2G20570.2            AtGLK1
 |  __|
 | |  |____________ Athalian AT5G44190.1            AtGLK2
 |_|
   |    , Ptrichoc Potri.007G136901.1     PtGLK-A
   |____|
        |_ Ptrichoc Potri.017G015800.2     PtGLK-B

```

Every tip's immediate sister is the same here as in the full 25-tip tree —
pruning did not create any of the groupings above. Verified before writing this
file.

## Reading it correctly — three caveats

1. **Branch lengths carry the "recent duplication" claim.** The Poplar pair sits
   on tips of 0.053 and 0.102; the Arabidopsis pair on 0.429 and 0.671. The
   Poplar copies are far less diverged from each other than the Arabidopsis
   copies are, which is what "indistinguishable recent duplication" means.

2. **The Arabidopsis pair is not sister to the Poplar pair.** In the full tree
   the eudicot group holds 15 tips and splits into the Arabidopsis pair on one
   side and everything else — citrus, apple (x4), tomato (x2), soybean (x4) and
   Poplar — on the other. Poplar is nested inside that second group. The pruned
   view preserves that split; it does not assert any special Arabidopsis-Poplar
   affinity.

3. **Do not read the deepest splits as phylogeny.** The resolved gene tree places
   the grass clade outside *Amborella*, which is not species-tree congruent. Only
   the within-grass and within-eudicot groupings used here are well supported.

## The identity scores do NOT support the assignment

The `psi_exclusive_call` and `psi_score` columns of
`glk_paralog_assignment.tsv` record the PlantSEED ortholog call. They are kept
because they are the obvious thing to reach for, and because they **disagree
with the labels for Sorghum** — so it is worth stating plainly why they do not
settle anything.

The call derives from the Arabidopsis-to-species ortholog tables
(`data/orthologs/Ath-Sbi-Orthologs.tsv`, `Ath-Ptr-Orthologs.tsv` in PlantSEED).
Those tables are a fully connected 2x2 with near-flat scores:

    AT2G20570 - Sobic.003G002600  0.42     AT2G20570 - Potri.007G136901  0.48
    AT2G20570 - Sobic.010G096300  0.41     AT2G20570 - Potri.017G015800  0.48
    AT5G44190 - Sobic.003G002600  0.39     AT5G44190 - Potri.007G136901  0.47
    AT5G44190 - Sobic.010G096300  0.38     AT5G44190 - Potri.017G015800  0.47

Every Arabidopsis GLK hits every species GLK. "Exclusive" means a 1:1
assignment was imposed on that graph after the fact — and for Sorghum the two
possible assignments **tie exactly** (0.42 + 0.38 = 0.41 + 0.39 = 0.80), so
which one the filter emitted is arbitrary. The other one would have paired
AtGLK2 with `Sobic.003G002600` and made the numbering appear to agree. For
Poplar all four scores fall within 0.01, which is the same indistinguishability
seen from the other side in the A/B pair.

All four values are also below the PlantSEED acceptance thresholds, monocot
0.55 and eudicot 0.60. They were accepted not on identity but because
OrthoFinder called the pairs orthologous — the `tag O` flag in the source
table, which is honoured regardless of score.

So the assignment rests on the gene tree and the grass literature. Nothing in
the identity scores contributes to it.

### One correction on record

The Sorghum slots were swapped on 2026-08-04. Before that, the figure inherited
the exclusive filter's arbitrary pick, which had `Sobic.003G002600` in the GLK1
slot. Since the reported +0.86 / +0.45 separation is attributed to the grass
GLK1, the labels had to follow the grass clades, not the tie-broken Arabidopsis
call.

## Files

| | |
|---|---|
| `glk_paralog_assignment.tsv` | the six focal genes, one row each |
| `glk_gene_tree_focal.nwk` | the 12-tip pruned tree, Newick |
| `glk_gene_tree_focal.txt` | the same tree, rendered (reproduced above) |

Columns of `glk_paralog_assignment.tsv`:

| | |
|---|---|
| `paper_label` | the name used in the manuscript and in `CONFIG.glk_orthologs` |
| `gene_tree_clade` | which grass clade the gene falls in, or why it has none |
| `psi_exclusive_call` / `psi_score` | the PlantSEED ortholog call — see the section above for why this is not the evidence |
| `immediate_sister_in_tree` | the gene's immediate sister in the full 25-tip tree |
| `tip_branch_length` | from the resolved gene tree; this is what carries the recent-duplication claim for Poplar |

## Provenance

OrthoFinder reference run, 15 species, 2021-05-26
(`/scratch1/seaver/OrthoFinder_Reference/OrthoFinder/Results_May26`).
Orthogroup OG0003894 holds all six focal GLKs plus maize, rice, *Brachypodium*,
*Amborella*, *Spirodela*, citrus, apple, tomato and soybean copies.
`glk_gene_tree_focal.nwk` is that orthogroup's **resolved** gene tree, pruned to
the twelve tips above with branch lengths preserved. The full 25-tip tree, the
unresolved tree, the MSA and the orthogroup sequences are not redistributed
here — they are OrthoFinder outputs, recoverable from the run above, and none
of the statements in this file depend on anything beyond the pruned tree except
the two whole-tree checks noted in the caveats, which were run against the full
tree before it was dropped.

The naming used in the figures is set by `CONFIG.glk_orthologs` in
`Paper_Figures/fig_photo_etc.py`.

Grass numbering follows the maize G2/ZmGLK1 and rice OsGLK1/OsGLK2 literature;
Arabidopsis numbering follows Waters et al. (2008, 2009). The two lineages were
numbered independently, before the relationships across the monocot/eudicot
split were resolved, which is why the digits cross.

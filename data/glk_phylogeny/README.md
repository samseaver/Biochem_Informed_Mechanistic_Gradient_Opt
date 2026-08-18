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

The grass and eudicot GLK duplications are **independent**. There is no 1:1
orthology between the grass GLK1/GLK2 pair and the Arabidopsis GLK1/GLK2 pair,
so an Arabidopsis-derived label cannot name a grass paralog. The Sorghum names
therefore come from **which grass clade each gene falls in**, not from which
Arabidopsis gene it scores best against.

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

`GLK_orthologs_exclusive.tsv` records the PlantSEED ortholog call. It is
included for completeness and because it is what the figure code consumes — but
it is not the evidence, and it disagrees with the labels for Sorghum. The
underlying score matrix is a fully connected 2x2 with near-flat values:

    AT2G20570 - Sobic.003G002600  0.42     AT2G20570 - Potri.007G136901  0.48
    AT2G20570 - Sobic.010G096300  0.41     AT2G20570 - Potri.017G015800  0.48
    AT5G44190 - Sobic.003G002600  0.39     AT5G44190 - Potri.007G136901  0.47
    AT5G44190 - Sobic.010G096300  0.38     AT5G44190 - Potri.017G015800  0.47

For Sorghum the two possible 1:1 assignments tie exactly
(0.42 + 0.38 = 0.41 + 0.39 = 0.80), so which one the exclusive filter emitted is
arbitrary. For Poplar all four scores fall within 0.01. All four values are also
below the monocot (0.55) and eudicot (0.60) acceptance thresholds — they were
accepted because OrthoFinder called the pairs orthologous, not on identity.

## Files

| | |
|---|---|
| `glk_paralog_assignment.tsv` | the six focal genes: label, clade, tree sister, tip branch length, and the PSI call |
| `glk_gene_tree_focal.nwk` | the 12-tip pruned tree, Newick |
| `glk_gene_tree_focal.txt` | the same tree, rendered (reproduced above) |
| `GLK_orthologs_exclusive.tsv` | the PlantSEED ortholog call consumed by `Paper_Figures/fig_photo_etc.py` |
| `orthofinder_OG0003894/` | the unmodified OrthoFinder extraction the above derives from |

## Provenance

OrthoFinder reference run, 15 species, 2021-05-26
(`/scratch1/seaver/OrthoFinder_Reference/OrthoFinder/Results_May26`).
Orthogroup OG0003894 holds all six focal GLKs plus maize, rice, *Brachypodium*,
*Amborella*, *Spirodela*, citrus, apple, tomato and soybean copies.
`orthofinder_OG0003894/` contains that orthogroup's resolved gene tree,
unresolved gene tree, MSA, sequences and membership row, copied unmodified.

The naming used in the figures is set by `CONFIG.glk_orthologs` in
`Paper_Figures/fig_photo_etc.py`.

Grass numbering follows the maize G2/ZmGLK1 and rice OsGLK1/OsGLK2 literature;
Arabidopsis numbering follows Waters et al. (2008, 2009). The two lineages were
numbered independently, before the relationships across the monocot/eudicot
split were resolved, which is why the digits cross.

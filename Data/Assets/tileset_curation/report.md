# Tileset Curation Report

## Summary
- Source: `Data\Assets\tileset_curation/tiles_curation.json`
- Base base tiles total: **22630**
- Unique tiles: **3052**
- Clusters (k): **20**

## Top 10 most frequent tiles
- 7371e25fe9e7a4a215a10f37d345768a6d5861f9: 2880 occurrences
- 044b60c9aaf09601abbd23a1b9539f95fcdd1107: 1934 occurrences
- ea8bce78aac3d800a6ac6ef9c57abe28fd84679b: 862 occurrences
- 059551b01910f858077d4cd7b743084f4c112d27: 716 occurrences
- 73da2daad18f00b04fea45d3f817f8ad3f83c9f9: 417 occurrences
- 10e607d5c68f5f58cae4ca48114ad0d9ade94d93: 374 occurrences
- d0a9324516fd5c56219ec7d35617163fa63c6cb6: 374 occurrences
- 3446947ae9e2567ae1647baa1b5d5e04d5a52d51: 367 occurrences
- 73f230bb9af0ee39384b4ebef13d0eb407ee27ec: 358 occurrences
- bbb9e4c044f55d7bf6548894110b739a36cd760c: 354 occurrences

## Coverage (frequency-based)
- Tiles to cover 90% of appearances: **833** (coverage 0.900)
- Tiles to cover 95% of appearances: **1921** (coverage 0.950)
- Tiles to cover 99% of appearances: **2826** (coverage 0.990)

## Heuristic tagging (brief)
- **blank/empty**: low variance or alpha transparency; high confidence
- **floor**: low edge strength and uniform color; medium confidence
- **wall/edge/corner**: detected via edge magnitude and side responses; medium confidence
- **door**: vertical high-contrast strip near center; medium confidence
- **stairs**: repeated horizontal edges, step-like profiles; medium confidence
- **deco**: many colors / high color variance; medium confidence
- **item/sprite**: small high-contrast blob(s) on uniform background; medium confidence
- **water**: blue-dominant mean color; medium confidence
- **rope**: thin vertical dark lines with brownish mean; low confidence

## Recommended atlas construction plan
- **Base tile size**: 8×8 (matches sheet)
- **Meta-tile strategy**: use 16×16 (2×2 base tiles) assembled from base tiles; include variants as needed for edges/corners
- **Suggested atlas dims**: **512×384** pixels (64×48 tiles of 8×8, capacity 3072 tiles) — power-of-two width and conservative square-ish height
- **Suggested final tile set**: include the top **1921** tiles to cover ~95% of on-screen usage by frequency, plus 1–2 representatives from each cluster to cover rare but visually distinct elements (estimated final count ~1941)
- **Ordering strategy**: order by frequency (most frequent first) then by group id to keep visually-related tiles near each other, which simplifies runtime lookup and compression

## Notes and next steps
- Run a quick manual review UI to confirm tag assignments and adjust `--k` to see cluster granularity changes
- Consider adding palette-aware deduplication for tiles that are identical up to palette swaps
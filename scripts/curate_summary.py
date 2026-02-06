import json
from pathlib import Path
p = Path('Data/Assets/tileset_curation')
tiles = json.load(open(p / 'tiles_curation.json'))['tiles']
groups = json.load(open(p / 'groups.json'))['groups']
sorted_tiles = sorted(tiles.items(), key=lambda kv: kv[1]['freq'], reverse=True)
total=sum(t['freq'] for t in tiles.values())
print('n_base_tiles_total (sum freqs):', total)
print('n_unique:', len(tiles))
print('top 10 tiles (uid, freq):')
for uid, info in sorted_tiles[:10]:
    print(uid, info['freq'])
print('n_groups:', len(groups))
# show group counts top 5
group_counts = sorted(((g, info['count']) for g, info in groups.items()), key=lambda x: x[1], reverse=True)
print('Top 5 groups by count:')
for g, c in group_counts[:5]:
    print(g, c)

# coverage stats
cs=0
r90=None
r95=None
r99=None
for i,(uid,info) in enumerate(sorted_tiles,1):
    cs+=info['freq']
    if r90 is None and cs/total>=0.9:
        r90=(i,cs/total)
    if r95 is None and cs/total>=0.95:
        r95=(i,cs/total)
    if r99 is None and cs/total>=0.99:
        r99=(i,cs/total)
print('\ncoverage thresholds:')
print('tiles to cover 90%:', r90)
print('tiles to cover 95%:', r95)
print('tiles to cover 99%:', r99)

# create concise markdown report
report = []
report.append('# Tileset Curation Report')
report.append('')
report.append('## Summary')
report.append(f'- Source: `{p}/tiles_curation.json`')
report.append(f'- Base base tiles total: **{total}**')
report.append(f'- Unique tiles: **{len(tiles)}**')
report.append(f'- Clusters (k): **{len(groups)}**')
report.append('')
report.append('## Top 10 most frequent tiles')
for uid, info in sorted_tiles[:10]:
    report.append(f'- {uid}: {info["freq"]} occurrences')
report.append('')
report.append('## Coverage (frequency-based)')
report.append(f'- Tiles to cover 90% of appearances: **{r90[0]}** (coverage {r90[1]:.3f})')
report.append(f'- Tiles to cover 95% of appearances: **{r95[0]}** (coverage {r95[1]:.3f})')
report.append(f'- Tiles to cover 99% of appearances: **{r99[0]}** (coverage {r99[1]:.3f})')
report.append('')
report.append('## Heuristic tagging (brief)')
report.append('- **blank/empty**: low variance or alpha transparency; high confidence')
report.append('- **floor**: low edge strength and uniform color; medium confidence')
report.append('- **wall/edge/corner**: detected via edge magnitude and side responses; medium confidence')
report.append('- **door**: vertical high-contrast strip near center; medium confidence')
report.append('- **stairs**: repeated horizontal edges, step-like profiles; medium confidence')
report.append('- **deco**: many colors / high color variance; medium confidence')
report.append('- **item/sprite**: small high-contrast blob(s) on uniform background; medium confidence')
report.append('- **water**: blue-dominant mean color; medium confidence')
report.append('- **rope**: thin vertical dark lines with brownish mean; low confidence')
report.append('')
report.append('## Recommended atlas construction plan')
report.append('- **Base tile size**: 8×8 (matches sheet)')
report.append('- **Meta-tile strategy**: use 16×16 (2×2 base tiles) assembled from base tiles; include variants as needed for edges/corners')
report.append(f'- **Suggested atlas dims**: **512×384** pixels (64×48 tiles of 8×8, capacity 3072 tiles) — power-of-two width and conservative square-ish height')
report.append(f'- **Suggested final tile set**: include the top **{r95[0]}** tiles to cover ~95% of on-screen usage by frequency, plus 1–2 representatives from each cluster to cover rare but visually distinct elements (estimated final count ~{r95[0]+len(groups)})')
report.append('- **Ordering strategy**: order by frequency (most frequent first) then by group id to keep visually-related tiles near each other, which simplifies runtime lookup and compression')
report.append('')
report.append('## Notes and next steps')
report.append('- Run a quick manual review UI to confirm tag assignments and adjust `--k` to see cluster granularity changes')
report.append('- Consider adding palette-aware deduplication for tiles that are identical up to palette swaps')
open(p / 'report.md', 'w', encoding='utf-8').write('\n'.join(report))
print('\nWrote report to', p / 'report.md')

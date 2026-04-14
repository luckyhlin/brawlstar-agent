# Data Sources Guide

## Video Sources

### Best for clean gameplay (no overlays)
- **Brawl Stars Esports** channel: tournament spectator views, clean game UI
  - https://www.youtube.com/@BrawlStarsEsports
  - Search: "brawl stars championship finals"
- **Mobile game recording** channels that post raw gameplay
  - Search: "brawl stars gameplay no commentary"
  - Search: "brawl stars ranked no facecam"

### Usable with cropping (have overlays)
- Most gameplay YouTubers have facecam in a corner
- The game area is typically a consistent rectangle
- Use `review-frames.py --crop` to define the game region once per video

### Search queries that work well
```
brawl stars gameplay no commentary 2026
brawl stars ranked match no commentary
brawl stars pro gameplay raw
brawl stars esports spectator view
brawl stars championship finals replay
brawl stars gem grab gameplay
brawl stars showdown no facecam
brawl stars brawl ball pro gameplay
```

## Character Reference Images

### Primary sources
1. **The Spriters Resource** — extracted game sprites
   - Trophy Road Brawler Portraits: https://www.spriters-resource.com/mobile/brawlstars/sheet/161823/
   - Chromatic Brawler Portraits: https://www.spriters-resource.com/mobile/brawlstars/sheet/161824/
   - In-Game Controls: https://www.spriters-resource.com/mobile/brawlstars/sheet/161859/
   - In-Game Indicators: https://www.spriters-resource.com/mobile/brawlstars/sheet/163271/

2. **Brawl Stars Fandom Wiki** — comprehensive per-character pages
   - https://brawlstars.fandom.com/wiki/Category:Brawlers
   - Has: portrait, model render, skins, attack animations

3. **Brawlify** — community database
   - https://brawlify.com/brawlers
   - Has: portraits, stats, tier info

4. **Pocket Tactics** — organized by class
   - https://www.pockettactics.com/brawl-stars/brawlers
   - 101 brawlers across 7 classes

### View transformation challenge
- **Reference images**: front-facing portraits, 3D model renders
- **In-game view**: 2.5D isometric/bird's-eye view
- Characters appear smaller, at an angle, with shadows and effects
- Need to build a mapping between reference portraits and in-game appearances
- Template matching alone won't work — need learned embeddings or feature matching

### Recommended approach
1. Download portraits as a reference catalog (run `fetch-character-refs.py`)
2. Manually screenshot each brawler in-game for ground truth
3. Build a small labeled dataset mapping portrait → in-game appearance
4. Use this for training a character classifier or embedding model

## Data Quality Checklist

For each video clip, verify:
- [ ] Resolution: 1080p or higher
- [ ] Contains actual gameplay (not just menus/intros)
- [ ] Game mode is identifiable (Gem Grab, Showdown, Brawl Ball, etc.)
- [ ] UI elements visible (health bars, timer, scores)
- [ ] No excessive overlays blocking game area
- [ ] If overlays present, game region is croppable

Use `review-frames.py --crop --sample 20` for quick verification.

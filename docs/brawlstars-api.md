# Brawl Stars Official API Reference

> Base URL: `https://api.brawlstars.com/v1`
> Docs: https://developer.brawlstars.com

## Authentication

All requests require a Bearer token in the `Authorization` header.

```bash
curl -H 'Authorization: Bearer <API_KEY>' https://api.brawlstars.com/v1/...
```

- API key is stored in `api.env` (git-ignored) as `BRAWL_STAR_API`
- Keys are IP-locked — generated at https://developer.brawlstars.com for a specific IP
- There is a **rate limit** on API token usage (exact threshold undocumented, but aggressive scanning will hit it)

## Player Tags

Player tags start with `#` and contain uppercase alphanumeric characters (e.g. `#RYY9LJVL`).
When used in URLs, `#` must be URL-encoded as `%23`:

```
In-game tag:  #RYY9LJVL
In URL:       %23RYY9LJVL
```

Our main account tag is stored in `api.env` as `MAJOR_ACCOUNT_TAG`.

---

## Error Responses

All error responses follow this shape:

```json
{
  "reason": "notFound",
  "message": "Not found with tag sij3f"
}
```

| Field   | Type   | Description                                      |
|---------|--------|--------------------------------------------------|
| reason  | string | Error code: `notFound`, `accessDenied`, `badRequest`, `throttled` |
| message | string | Human-readable error description                 |
| type    | string | (optional) Error category                        |
| detail  | object | (optional) Additional error context               |

---

## Endpoints

### 1. GET `/players/{playerTag}/battlelog`

**Get recent battle log for a player.**

Returns up to 25 recent battles. New battles may take up to 30 minutes to appear.

#### Parameters

| Name      | In   | Type   | Required | Description        |
|-----------|------|--------|----------|--------------------|
| playerTag | path | string | yes      | Player tag (URL-encoded) |

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/players/%23RYY9LJVL/battlelog'
```

#### Response Schema

```
BattleList {
  items: Battle[]
  paging: { cursors: {} }
}
```

Each `Battle` has two shapes depending on the game mode:

**Team modes** (gemGrab, brawlBall, heist, bounty, knockout, hotZone, etc.):

| Field               | Type     | Description                                   |
|---------------------|----------|-----------------------------------------------|
| battleTime          | string   | ISO-ish timestamp: `YYYYMMDDTHHmmSS.000Z`    |
| event.id            | integer  | Unique event ID                               |
| event.mode          | string   | Mode name (e.g. `gemGrab`, `knockout`)        |
| event.modeId        | integer  | Numeric mode ID                               |
| event.map           | string   | Map name (e.g. `"Out in the Open"`)           |
| battle.mode         | string   | Same as event.mode                            |
| battle.type         | string   | `ranked`, `soloRanked`, `friendly`, etc.      |
| battle.result       | string   | `victory` or `defeat`                         |
| battle.duration     | integer  | Match duration in seconds                     |
| battle.trophyChange | integer  | (optional) Trophy delta, present in `ranked` type |
| battle.starPlayer   | Player   | (optional) MVP of the match                   |
| battle.teams        | Player[][] | 2 teams, each with 3 players                |

**Solo Showdown** (soloShowdown):

| Field               | Type     | Description                              |
|---------------------|----------|------------------------------------------|
| battle.mode         | string   | `soloShowdown`                           |
| battle.type         | string   | `ranked`                                 |
| battle.rank         | integer  | Final placement (1 = winner, 10 = last)  |
| battle.trophyChange | integer  | Trophy delta                             |
| battle.players      | Player[] | Flat list of all 10 players              |

**Player object** (within battle):

| Field           | Type    | Description                |
|-----------------|---------|----------------------------|
| tag             | string  | Player tag with `#`        |
| name            | string  | Display name               |
| brawler.id      | integer | Brawler ID (e.g. 16000001) |
| brawler.name    | string  | Brawler name (UPPERCASE)   |
| brawler.power   | integer | Power level (1-11)         |
| brawler.trophies| integer | Brawler trophies at time of match |

#### Key Observations (from live data)

- Log contains exactly **25 battles** (API maximum)
- `battleTime` format: `20260413T185800.000Z` — not standard ISO 8601 (no dashes/colons)
- `battle.type` values seen: `ranked` (ladder with trophy change), `soloRanked` (ranked mode without trophy change, trophies show as 7-8)
- `event.mode` can be `"unknown"` for newer modes not yet in the API enum (e.g. modeId 45 = Brawl Hockey)
- `starPlayer` is absent in Showdown modes
- `trophyChange` is absent in `soloRanked` type battles
- Team arrays: `teams[0]` and `teams[1]` — the player can be on either team

---

### 2. GET `/players/{playerTag}`

**Get player profile information.**

#### Parameters

| Name      | In   | Type   | Required | Description        |
|-----------|------|--------|----------|--------------------|
| playerTag | path | string | yes      | Player tag (URL-encoded) |

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/players/%23RYY9LJVL'
```

#### Response Schema — Player

| Field                              | Type    | Description                          |
|------------------------------------|---------|--------------------------------------|
| tag                                | string  | Player tag                           |
| name                               | string  | Display name                         |
| nameColor                          | string  | Hex color string (e.g. `0xffa8e132`) |
| icon.id                            | integer | Profile icon ID                      |
| trophies                           | integer | Current total trophies               |
| highestTrophies                    | integer | All-time highest trophies            |
| totalPrestigeLevel                 | integer | Sum of prestige across all brawlers  |
| expLevel                           | integer | Experience level                     |
| expPoints                          | integer | Total experience points              |
| isQualifiedFromChampionshipChallenge | boolean | Championship qualification status  |
| 3vs3Victories                      | integer | Total 3v3 wins                       |
| soloVictories                      | integer | Total solo showdown wins             |
| duoVictories                       | integer | Total duo showdown wins              |
| bestRoboRumbleTime                 | integer | Best Robo Rumble time (minutes)      |
| bestTimeAsBigBrawler               | integer | Best Big Game survival time          |
| club.tag                           | string  | Club tag (if in a club)              |
| club.name                          | string  | Club name (if in a club)             |
| brawlers                           | array   | All brawlers the player owns         |

#### Player's Brawler object (richer than battlelog version)

| Field            | Type    | Description                                |
|------------------|---------|--------------------------------------------|
| id               | integer | Brawler ID                                 |
| name             | string  | Brawler name (UPPERCASE)                   |
| power            | integer | Power level (1-11)                         |
| rank             | integer | Brawler rank                               |
| trophies         | integer | Current trophies on this brawler           |
| highestTrophies  | integer | All-time highest trophies on this brawler  |
| prestigeLevel    | integer | Prestige level for this brawler            |
| currentWinStreak | integer | Current consecutive wins                   |
| maxWinStreak     | integer | Best ever win streak                       |
| skin.id          | integer | Equipped skin ID                           |
| skin.name        | string  | Equipped skin name                         |
| gadgets          | array   | Unlocked gadgets `[{id, name}]`            |
| gears            | array   | Equipped gears `[{id, name, level}]`       |
| starPowers       | array   | Unlocked star powers `[{id, name}]`        |
| hyperCharges     | array   | Unlocked hyper charges `[{id, name}]`      |
| buffies.gadget     | boolean | Whether gadget slot is available/unlocked |
| buffies.starPower  | boolean | Whether star power slot is available      |
| buffies.hyperCharge| boolean | Whether hyper charge is available         |

#### Key Observations

- Response is **very large** (~100KB+) because it includes all 101 brawlers with full loadout details
- Main account ("Call Me Dad", `#RYY9LJVL`): 29,136 trophies, 4,636 3v3 wins, level 115
- Useful for mapping brawler IDs to names, checking loadouts, tracking progression

---

### 3. GET `/rankings/{countryCode}/players`

**Get player leaderboard (country or global).**

#### Parameters

| Name        | In    | Type    | Required | Description                                     |
|-------------|-------|---------|----------|-------------------------------------------------|
| countryCode | path  | string  | yes      | Two-letter country code or `global`             |
| before      | query | string  | no       | Pagination cursor (from `paging.cursors.before`) |
| after       | query | string  | no       | Pagination cursor (from `paging.cursors.after`)  |
| limit       | query | integer | no       | Max items to return (default 200)               |

Only `before` OR `after` can be specified, not both.

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/rankings/global/players?limit=5'
```

#### Response Schema

```
PlayerRankingList {
  items: PlayerRanking[]
  paging: { cursors: { after?: string, before?: string } }
}
```

| Field         | Type    | Description                          |
|---------------|---------|--------------------------------------|
| tag           | string  | Player tag                           |
| name          | string  | Display name                         |
| nameColor     | string  | Hex color (e.g. `0xffff8afb`)        |
| icon.id       | integer | Profile icon ID                      |
| trophies      | integer | Current total trophies               |
| rank          | integer | Leaderboard position (1-indexed)     |
| club.name     | string  | (optional) Club name                 |

#### Use Case

Primary way to **discover player tags** for analysis without guessing. The global top 200 provides a good seed set of active high-level players. Pagination via `after` cursor allows fetching beyond the first page.

---

### 4. GET `/brawlers`

**Get list of all available brawlers and their equipment.**

#### Parameters

| Name   | In    | Type    | Required | Description         |
|--------|-------|---------|----------|---------------------|
| before | query | string  | no       | Pagination cursor   |
| after  | query | string  | no       | Pagination cursor   |
| limit  | query | integer | no       | Max items to return |

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/brawlers?limit=3'
```

#### Response Schema

```
BrawlerList {
  items: Brawler[]
  paging: { cursors: { after?: string } }
}
```

| Field        | Type  | Description                              |
|--------------|-------|------------------------------------------|
| id           | integer | Brawler ID (16000000–16000103)         |
| name         | string  | Brawler name (UPPERCASE)               |
| starPowers   | array   | Available star powers `[{id, name}]`   |
| hyperCharges | array   | Available hyper charges `[{id, name}]` |
| gears        | array   | Available gears `[{id, name, level}]`  |
| gadgets      | array   | Available gadgets `[{id, name}]`       |

#### Key Observations

- Currently **101 brawlers** (SHELLY id=16000000 through NAJIA id=16000103, with gaps)
- This is the canonical source of brawler IDs for cross-referencing with battlelog data
- Gears are shared across brawlers: SPEED, HEALTH, DAMAGE, VISION, SHIELD, RELOAD SPEED, SUPER CHARGE, GADGET COOLDOWN
- No stats (HP, damage, range) are exposed — only equipment metadata

---

### 5. GET `/brawlers/{brawlerId}`

**Get information about a single brawler.**

#### Parameters

| Name      | In   | Type   | Required | Description             |
|-----------|------|--------|----------|-------------------------|
| brawlerId | path | string | yes      | Numeric brawler ID      |

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/brawlers/16000001'
```

#### Response

Same schema as a single item from the `/brawlers` list (id, name, starPowers, hyperCharges, gears, gadgets). Useful when you only need one brawler's data.

---

### 6. GET `/gamemodes`

**Get list of all known game modes.**

#### Parameters

| Name   | In    | Type    | Required | Description         |
|--------|-------|---------|----------|---------------------|
| before | query | string  | no       | Pagination cursor   |
| after  | query | string  | no       | Pagination cursor   |
| limit  | query | integer | no       | Max items to return |

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/gamemodes'
```

#### Response Schema

```
EventTypeList {
  items: EventType[]
  paging: { cursors: {} }
}
```

| Field | Type    | Description              |
|-------|---------|--------------------------|
| id    | integer | Mode ID                  |
| name  | string  | Mode name (UPPERCASE) or `null` for unreleased modes |

#### Mode ID Reference Table (from live data, 2026-04-13)

| ID | Name                 | ID | Name                  |
|----|----------------------|----|-----------------------|
| 0  | GEM GRAB             | 26 | PAYLOAD               |
| 2  | HEIST                | 27 | BOT DROP              |
| 3  | BOUNTY               | 28 | HUNTERS               |
| 5  | BRAWL BALL           | 31 | WIPEOUT 5V5           |
| 6  | SOLO SHOWDOWN        | 32 | BRAWL BALL 5V5        |
| 7  | BIG GAME             | 33 | GEM GRAB 5V5          |
| 8  | ROBO RUMBLE          | 34 | TROPHY ESCAPE         |
| 9  | DUO SHOWDOWN         | 35 | KNOCKOUT 5V5          |
| 10 | BOSS FIGHT           | 37 | PAINT BRAWL           |
| 11 | SPIRIT WARS          | 38 | TRIO SHOWDOWN         |
| 14 | TAKEDOWN             | 40 | SOUL COLLECTOR        |
| 16 | PRESENT PLUNDER      | 41 | CLEANING DUTY         |
| 17 | HOT ZONE             | 45 | BRAWL HOCKEY          |
| 20 | KNOCKOUT             | 46 | GEM GRAB 2V2          |
| 21 | CARRY THE GIFT       | 47 | SPECIAL DELIVERY      |
| 22 | BASKET BRAWL         | 48 | BRAWL ARENA           |
| 23 | VOLLEY BRAWL         | 49-54 | Various 2V2 modes  |
| 24 | DUELS                | 55 | TOKEN RUN             |
| 25 | WIPEOUT              | 56-74 | Seasonal/event modes |

---

### 7. GET `/events/rotation`

**Get currently active event rotation.**

#### Parameters

None.

#### Example

```bash
curl -H 'Authorization: Bearer $API_KEY' \
  'https://api.brawlstars.com/v1/events/rotation'
```

#### Response Schema

Returns a **plain JSON array** (not wrapped in `{items: [...]}`).

```
ScheduledEvent[] (top-level array)
```

| Field           | Type    | Description                                 |
|-----------------|---------|---------------------------------------------|
| startTime       | string  | Rotation start: `YYYYMMDDTHHmmSS.000Z`     |
| endTime         | string  | Rotation end: `YYYYMMDDTHHmmSS.000Z`       |
| slotId          | integer | Slot number (determines UI position)        |
| event.id        | integer | Unique event ID                             |
| event.mode      | string  | Mode name (lowercase, e.g. `brawlBall`)     |
| event.modeId    | integer | Numeric mode ID                             |
| event.map       | string  | Map name                                    |

#### Key Observations

- Rotations are typically **24 hours** each (some special modes shorter, e.g. 2 hours)
- `slotId` groups events into UI rows: 1-6 are the main slots, higher numbers are special/rotating
- `event.mode` can be `"unknown"` for modes newer than the API enum
- Timestamps use the same non-ISO format as battlelog: `20260413T080000.000Z`

#### Mode Enum Values (from API schema)

soloShowdown, duoShowdown, heist, bounty, siege, gemGrab, brawlBall, bigGame, bossFight,
roboRumble, takedown, loneStar, presentPlunder, hotZone, superCityRampage, knockout,
volleyBrawl, basketBrawl, holdTheTrophy, trophyThieves, duels, wipeout, payload, botDrop,
hunters, lastStand, snowtelThieves, pumpkinPlunder, trophyEscape, wipeout5V5, knockout5V5,
gemGrab5V5, brawlBall5V5, godzillaCitySmash, paintBrawl, trioShowdown, zombiePlunder,
jellyfishing, duoMegaBoss, loveBombing, bombHeist, shadowSmash, shadowSmash5V5, unknown

#### Event Modifier Enum

unknown, none, energyDrink, angryRobo, meteorShower, graveyardShift, healingMushrooms,
bossFightRockets, takedownLasers, takedownChainLightning, takedownRockets, waves,
hauntedBall, superCharge, fastBrawlers, showdown+, peekABoo, burningBall

---

## Omitted Endpoints

These endpoints exist but are not documented here (not needed for current research scope):

- `GET /clubs/{clubTag}` — Club info
- `GET /clubs/{clubTag}/members` — Club members
- `GET /rankings/{countryCode}/clubs` — Club leaderboard
- `GET /rankings/{countryCode}/brawlers/{brawlerId}` — Per-brawler leaderboard

---

## ID Namespaces

| Prefix    | Entity        | Example                   |
|-----------|---------------|---------------------------|
| 16000xxx  | Brawler       | 16000001 = COLT           |
| 23000xxx  | Gadget / Star Power / Hyper Charge | 23000077 = SLICK BOOTS |
| 28000xxx  | Player Icon   | 28000777                  |
| 29000xxx  | Skin          | 29000029 = BANDITA SHELLY |
| 62000xxx  | Gear          | 62000002 = DAMAGE         |
| 15000xxx  | Event         | 15000548 = Knockout on "Out in the Open" |

---

## Timestamp Format

All timestamps use a compact format without separators:

```
20260413T185800.000Z
```

Parsing pattern: `YYYYMMDDTHHmmSS.sssZ` — always UTC.

Python parsing:

```python
from datetime import datetime
dt = datetime.strptime("20260413T185800.000Z", "%Y%m%dT%H%M%S.%fZ")
```

---

## Pagination

List endpoints use cursor-based pagination:

```json
{
  "items": [...],
  "paging": {
    "cursors": {
      "after": "eyJwb3MiOjV9"
    }
  }
}
```

- Pass `after` value as query parameter to get the next page
- If `cursors` is empty `{}`, there are no more pages
- `before` and `after` are mutually exclusive

---

## Rate Limiting

- Exact rate limits are not publicly documented
- The developer tier is "developer/silver" (based on JWT claims)
- Aggressive scanning (e.g. iterating random player tags) will get throttled
- Recommended: use rankings endpoint to discover tags, then fetch profiles/battlelogs

---

## Example Files

Full API responses from 2026-04-13 are saved in `docs/api-examples/`:

| File                              | Endpoint                            | Notes                        |
|-----------------------------------|-------------------------------------|------------------------------|
| `battlelog.json`                  | `/players/%23RYY9LJVL/battlelog`    | 25 recent battles            |
| `player.json`                     | `/players/%23RYY9LJVL`              | Full profile (101 brawlers)  |
| `rankings-global-players.json`    | `/rankings/global/players?limit=5`  | Top 5 global                 |
| `rankings-global-players-full.json` | `/rankings/global/players?limit=200` | Top 200 global             |
| `brawlers-list.json`             | `/brawlers?limit=3`                 | First 3 brawlers             |
| `brawlers-all.json`              | `/brawlers`                         | All 101 brawlers             |
| `brawler-single.json`            | `/brawlers/16000001`                | COLT details                 |
| `gamemodes.json`                  | `/gamemodes`                        | All 57 game modes            |
| `events-rotation.json`           | `/events/rotation`                  | Current rotation (13 events) |

---

## Quick Reference: Battle Log Variants

The battlelog response shape differs by mode category:

### Team 3v3 (gemGrab, brawlBall, heist, bounty, knockout, hotZone)

```
battle.result       → "victory" | "defeat"
battle.duration     → seconds
battle.starPlayer   → Player (MVP)
battle.trophyChange → int (only in "ranked" type)
battle.teams        → [[Player, Player, Player], [Player, Player, Player]]
```

### Solo Showdown

```
battle.rank         → 1-10 (placement)
battle.trophyChange → int
battle.players      → [Player × 10] (flat list, ordered by placement)
```

### Duo Showdown (expected, not in current log)

```
battle.rank         → 1-5 (team placement)
battle.trophyChange → int
battle.teams        → [[Player, Player] × 5]
```

### Ranked Types

- `"ranked"` — Trophy-based ladder, shows `trophyChange` (can be negative)
- `"soloRanked"` — Competitive mode, trophies show as 7-8 (rank points, not trophy trophies), no `trophyChange` field

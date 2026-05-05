#!/usr/bin/env python3
"""Local web dashboard for Brawl Stars battle analytics.

Serves an interactive HTML dashboard on localhost showing:
- Brawler win rates with portraits, filterable by game mode
- Team composition win rates
- Matchup matrix (brawler vs opposing brawler)
- Synergy matrix (brawler pairs on same team)
- Database summary stats

Usage:
    PYTHONPATH=src uv run python scripts/dashboard.py
    PYTHONPATH=src uv run python scripts/dashboard.py --port 8888
"""

import argparse
import base64
import json
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.analytics import BattleAnalytics
from brawlstar_agent.dashboard_data import (
    CACHE_PATH,
    collect_all_data,
    read_cache,
    write_cache,
)
from brawlstar_agent.db import DEFAULT_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTRAITS_DIR = PROJECT_ROOT / "datasets" / "character_refs" / "portraits"
BRAWLER_INDEX = PROJECT_ROOT / "datasets" / "character_refs" / "brawlers_index.json"


def load_portrait_map() -> dict[int, str]:
    """Map brawler_id -> base64 data URI for the borderless portrait."""
    portraits = {}
    if not PORTRAITS_DIR.exists():
        return portraits
    for png in PORTRAITS_DIR.glob("*_borderless.png"):
        try:
            brawler_id = int(png.stem.split("_")[0])
            b64 = base64.b64encode(png.read_bytes()).decode()
            portraits[brawler_id] = f"data:image/png;base64,{b64}"
        except (ValueError, IOError):
            continue
    return portraits


def load_brawler_name_map() -> dict[str, int]:
    """Map UPPERCASE brawler name -> brawler_id from the index.

    Builds aliases so API names like 'EL PRIMO' match index names like 'El-Primo'.
    """
    name_to_id = {}
    if BRAWLER_INDEX.exists():
        index = json.loads(BRAWLER_INDEX.read_text())
        for b in index:
            canonical = b["name"].upper()
            name_to_id[canonical] = b["id"]
            # Also map variants: hyphens <-> spaces, strip punctuation
            alt = canonical.replace("-", " ")
            name_to_id[alt] = b["id"]
            alt2 = canonical.replace("-", ". ").rstrip()
            name_to_id[alt2] = b["id"]
            alt3 = canonical.replace("-", " & ")
            name_to_id[alt3] = b["id"]

    # Direct ID-based map from the DB as final fallback
    import sqlite3
    try:
        conn = sqlite3.connect(str(DEFAULT_DB_PATH))
        for row in conn.execute("SELECT id, name FROM brawlers"):
            if row[1] not in name_to_id:
                name_to_id[row[1]] = row[0]
        conn.close()
    except Exception:
        pass

    return name_to_id


def generate_html(data: dict, portraits: dict[int, str], name_to_id: dict[str, int]) -> str:
    portrait_json = json.dumps({str(k): v for k, v in portraits.items()})
    name_id_json = json.dumps(name_to_id)
    data_json = json.dumps(data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brawl Stars Battle Analytics</title>
<style>
:root {{
    --bg: #0f1923;
    --card: #1a2634;
    --border: #2a3a4a;
    --text: #e0e6ed;
    --text-dim: #8899aa;
    --accent: #4fc3f7;
    --green: #66bb6a;
    --red: #ef5350;
    --yellow: #ffd54f;
    --orange: #ffa726;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
.subtitle {{ color: var(--text-dim); margin-bottom: 20px; }}
.stats-bar {{
    display: flex; gap: 20px; flex-wrap: wrap;
    margin-bottom: 24px; padding: 16px;
    background: var(--card); border-radius: 8px; border: 1px solid var(--border);
}}
.stat {{ text-align: center; min-width: 100px; }}
.stat-val {{ font-size: 1.6em; font-weight: 700; color: var(--accent); }}
.stat-label {{ font-size: 0.8em; color: var(--text-dim); text-transform: uppercase; }}

.tabs {{
    display: flex; gap: 4px; margin-bottom: 16px;
    border-bottom: 2px solid var(--border); padding-bottom: 0;
}}
.tab {{
    padding: 10px 20px; cursor: pointer; border: none; background: none;
    color: var(--text-dim); font-size: 0.95em; font-weight: 500;
    border-bottom: 3px solid transparent; transition: all 0.2s;
}}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.mode-filter {{
    display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px;
}}
.mode-btn {{
    padding: 6px 14px; border-radius: 16px; border: 1px solid var(--border);
    background: var(--card); color: var(--text-dim); cursor: pointer;
    font-size: 0.85em; transition: all 0.15s;
}}
.mode-btn:hover {{ border-color: var(--accent); color: var(--text); }}
.mode-btn.active {{ background: var(--accent); color: var(--bg); border-color: var(--accent); font-weight: 600; }}

.brawler-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px;
}}
.brawler-card {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; background: var(--card);
    border-radius: 8px; border: 1px solid var(--border);
    transition: border-color 0.15s;
}}
.brawler-card:hover {{ border-color: var(--accent); }}
.brawler-card img {{
    width: 40px; height: 40px; border-radius: 50%;
    background: #2a3a4a; object-fit: cover;
}}
.brawler-info {{ flex: 1; min-width: 0; }}
.brawler-name {{
    font-size: 0.8em; font-weight: 600;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.brawler-stats {{ font-size: 0.75em; color: var(--text-dim); }}
.win-rate {{
    font-size: 1.1em; font-weight: 700; white-space: nowrap;
}}
.wr-high {{ color: var(--green); }}
.wr-mid {{ color: var(--yellow); }}
.wr-low {{ color: var(--red); }}
.ci-bar {{
    position: relative; height: 4px; background: var(--border); border-radius: 2px;
    margin-top: 4px; overflow: hidden;
}}
.ci-fill {{
    position: absolute; height: 100%; border-radius: 2px; background: var(--accent); opacity: 0.5;
}}
.ci-mark {{
    position: absolute; height: 100%; width: 2px; background: var(--accent);
}}
.brawler-meta {{
    font-size: 0.7em; color: var(--text-dim); margin-top: 2px; line-height: 1.3;
}}
.sort-toggle {{
    display: inline-flex; gap: 4px; margin-bottom: 12px; margin-left: 8px;
    vertical-align: middle;
}}
.sort-btn {{
    padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border);
    background: none; color: var(--text-dim); cursor: pointer; font-size: 0.8em;
}}
.sort-btn:hover {{ color: var(--text); border-color: var(--accent); }}
.sort-btn.active {{ background: var(--accent); color: var(--bg); border-color: var(--accent); font-weight: 600; }}
.section-label {{
    margin-bottom: 8px; font-size: 0.8em; color: var(--text-dim);
    text-transform: uppercase; font-weight: 600;
}}

table {{
    width: 100%; border-collapse: collapse;
    background: var(--card); border-radius: 8px; overflow: hidden;
}}
th {{
    text-align: left; padding: 10px 12px;
    background: rgba(79, 195, 247, 0.1); color: var(--accent);
    font-size: 0.8em; text-transform: uppercase; font-weight: 600;
    position: sticky; top: 0;
}}
td {{ padding: 8px 12px; border-top: 1px solid var(--border); font-size: 0.9em; }}
tr:hover td {{ background: rgba(255,255,255,0.03); }}
.portrait-sm {{ width: 28px; height: 28px; border-radius: 50%; vertical-align: middle; margin-right: 6px; background: #2a3a4a; }}
.combo-portraits {{ display: flex; gap: 2px; align-items: center; }}
.combo-portraits img {{ width: 24px; height: 24px; border-radius: 50%; background: #2a3a4a; }}
.combo-plus {{ color: var(--text-dim); font-size: 0.8em; margin: 0 2px; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; border-radius: 8px; border: 1px solid var(--border); }}

.search-box {{
    padding: 8px 14px; background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 0.9em; width: 250px;
    margin-bottom: 12px;
}}
.search-box::placeholder {{ color: var(--text-dim); }}
.search-box:focus {{ outline: none; border-color: var(--accent); }}
</style>
</head>
<body>
<div class="container">
    <h1>Brawl Stars Battle Analytics</h1>
    <p class="subtitle" id="subtitle"></p>

    <div class="stats-bar" id="statsBar"></div>

    <div class="tabs">
        <button class="tab active" data-tab="brawlers">Brawler Win Rates</button>
        <button class="tab" data-tab="combos">Team Compositions</button>
        <button class="tab" data-tab="matchups">Matchups</button>
        <button class="tab" data-tab="synergies">Synergies</button>
        <button class="tab" data-tab="mydata">My Data</button>
    </div>

    <div id="brawlers" class="tab-content active">
        <div class="section-label">Game Mode</div>
        <div class="mode-filter" id="brawlerModeFilter"></div>
        <div class="section-label" style="margin-top:12px">Match Type &amp; Skill Tier</div>
        <div class="mode-filter" id="brawlerTypeFilter"></div>
        <div class="section-label" style="margin-top:12px">
            Sort by
            <span class="sort-toggle" id="sortToggle"></span>
            <span style="font-weight:400;margin-left:8px" id="sortDesc"></span>
        </div>
        <div class="brawler-grid" id="brawlerGrid"></div>
    </div>

    <div id="combos" class="tab-content">
        <div class="mode-filter" id="comboModeFilter"></div>
        <div class="table-wrap"><table id="comboTable"><thead><tr>
            <th>Composition</th><th>Win Rate</th><th>Wins</th><th>Games</th>
        </tr></thead><tbody></tbody></table></div>
    </div>

    <div id="matchups" class="tab-content">
        <input class="search-box" id="matchupSearch" placeholder="Filter by brawler name...">
        <div class="table-wrap"><table id="matchupTable"><thead><tr>
            <th>Brawler</th><th>vs Opponent</th><th>Win Rate</th><th>Games</th>
        </tr></thead><tbody></tbody></table></div>
    </div>

    <div id="synergies" class="tab-content">
        <input class="search-box" id="synergySearch" placeholder="Filter by brawler name...">
        <div class="table-wrap"><table id="synergyTable"><thead><tr>
            <th>Brawler A</th><th>Brawler B</th><th>Win Rate</th><th>Games</th>
        </tr></thead><tbody></tbody></table></div>
    </div>

    <div id="mydata" class="tab-content">
        <div id="myProfile"></div>
        <div class="section-label" style="margin-top:16px">My Brawler Stats</div>
        <div class="brawler-grid" id="myBrawlerGrid"></div>
        <div class="section-label" style="margin-top:16px">Win Rate by Mode</div>
        <div id="myModeStats" style="margin-bottom:16px"></div>
        <div class="section-label" style="margin-top:16px">Battle Log <span style="font-weight:400" id="myBattleCount"></span></div>
        <div class="mode-filter" id="myLogFilter"></div>
        <div class="table-wrap" style="max-height:800px"><table id="myLogTable"><thead><tr>
            <th>Time</th><th>Mode</th><th>Map</th><th>Brawler</th><th>Result</th><th>Teammates</th><th>Opponents</th>
        </tr></thead><tbody></tbody></table></div>
    </div>
</div>

<script>
const PORTRAITS = {portrait_json};
const NAME_TO_ID = {name_id_json};
const DATA = {data_json};

function portraitUrl(name) {{
    const id = NAME_TO_ID[name] || NAME_TO_ID[name.toUpperCase()];
    return id ? (PORTRAITS[String(id)] || '') : '';
}}

function wrClass(rate) {{
    if (rate >= 55) return 'wr-high';
    if (rate >= 45) return 'wr-mid';
    return 'wr-low';
}}

function portraitImg(name, cls) {{
    const url = portraitUrl(name);
    cls = cls || 'portrait-sm';
    return url ? `<img src="${{url}}" class="${{cls}}" alt="${{name}}">` : '';
}}

function prettyMode(m) {{
    return m.replace(/([A-Z])/g, ' $1').replace(/5 V 5/g, '5v5').replace(/^./, s => s.toUpperCase()).trim();
}}

// Stats bar
const s = DATA.summary;
const btd = s.battle_type_distribution || {{}};

function fmtAge(iso) {{
    if (!iso) return null;
    const t = new Date(iso);
    if (isNaN(t)) return null;
    const secs = (Date.now() - t.getTime()) / 1000;
    if (secs < 60) return Math.floor(secs) + 's ago';
    if (secs < 3600) return Math.floor(secs / 60) + ' min ago';
    if (secs < 86400) return (secs / 3600).toFixed(1) + ' h ago';
    return (secs / 86400).toFixed(1) + ' d ago';
}}

const sub = document.getElementById('subtitle');
sub.textContent =
    `${{s.total_battles.toLocaleString()}} battles from ${{s.total_players.toLocaleString()}} players` +
    ` (${{s.earliest_battle?.slice(0,10) || '?'}} to ${{s.latest_battle?.slice(0,10) || '?'}})`;

const meta = DATA._cache_meta;
if (meta && meta.computed_at) {{
    const ageStr = fmtAge(meta.computed_at);
    const ageSecs = (Date.now() - new Date(meta.computed_at).getTime()) / 1000;
    let color = 'var(--text-dim)';
    if (ageSecs > 86400) color = 'var(--red)';
    else if (ageSecs > 21600) color = 'var(--orange)';
    let computeNote = '';
    if (meta.computed_in_seconds != null) {{
        const cs = meta.computed_in_seconds;
        let computeColor = 'var(--text-dim)';
        if (cs > 2700) computeColor = 'var(--red)';
        else if (cs > 1800) computeColor = 'var(--orange)';
        const t = cs > 60 ? `${{(cs / 60).toFixed(1)}} min` : `${{cs.toFixed(1)}}s`;
        computeNote = ` · <span style="color:${{computeColor}}">compute took ${{t}}</span>`;
    }}
    sub.innerHTML = sub.textContent +
        `<br><span style="color:${{color}}">analytics cached ${{ageStr}}</span>${{computeNote}}`;
}} else {{
    sub.innerHTML = sub.textContent +
        `<br><span style="color:var(--orange)">computed inline (no cache) — run scripts/precompute-analytics.py to enable caching</span>`;
}}
const statsHtml = [
    ['Battles', s.total_battles],
    ['Ranked', btd['ranked'] || 0],
    ['Solo Ranked', btd['soloRanked'] || 0],
    ['Players', s.total_players],
    ['Brawlers', s.total_brawlers],
].map(([l,v]) => `<div class="stat"><div class="stat-val">${{v.toLocaleString()}}</div><div class="stat-label">${{l}}</div></div>`).join('');
document.getElementById('statsBar').innerHTML = statsHtml;

// Tabs
document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
    }});
}});

// Wilson interval (client-side, for sort on any view)
function wilsonLower(wins, total, z) {{
    z = z || 1.96;
    if (total === 0) return 0;
    const p = wins / total;
    const z2 = z * z;
    const d = 1 + z2 / total;
    const c = (p + z2 / (2 * total)) / d;
    const s = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total) / d;
    return Math.max(0, c - s);
}}

// Brawler win rates
let currentBrawlerMode = 'all';
let currentBrawlerType = 'all';
let currentSort = 'raw_wr';
const sortOptions = [
    ['raw_wr', 'Raw WR', 'Win rate = wins / total games'],
    ['adjusted_wr', 'Adjusted WR', 'Tier-standardized, removes skill-mix bias'],
    ['wilson_lower', 'Confidence', 'Wilson lower bound, penalizes small samples'],
];
const tierLabels = {{
    'all': 'All Types',
    'ladder_all': 'Ladder (all)',
    'competitive_all': 'Competitive (all)',
}};
// Ladder trophy tiers
(DATA.trophy_tiers || []).forEach(t => {{
    const hi = t.hi > 9999 ? '∞' : t.hi;
    tierLabels['ladder_' + t.name] = '🏆 ' + t.name + ' (' + t.lo + '-' + hi + ')';
}});
// Competitive ranked tiers
(DATA.ranked_tiers || []).forEach(t => {{
    tierLabels['competitive_' + t.name] = '⚔ ' + t.name;
}});
const rankedTierNames = (DATA.ranked_tiers || []).map(t => t.name);

function renderSortToggle() {{
    const container = document.getElementById('sortToggle');
    container.innerHTML = sortOptions.map(([k, label]) =>
        `<button class="sort-btn ${{k === currentSort ? 'active' : ''}}" data-sort="${{k}}">${{label}}</button>`
    ).join('');
    container.querySelectorAll('.sort-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
            currentSort = btn.dataset.sort;
            renderSortToggle();
            renderBrawlerGrid();
        }});
    }});
    const desc = sortOptions.find(s => s[0] === currentSort);
    document.getElementById('sortDesc').textContent = desc ? desc[2] : '';
}}

function renderBrawlerModes() {{
    const modes = ['all', ...DATA.modes];
    const container = document.getElementById('brawlerModeFilter');
    container.innerHTML = modes.map(m =>
        `<button class="mode-btn ${{m === currentBrawlerMode ? 'active' : ''}}" data-mode="${{m}}">${{m === 'all' ? 'All Modes' : prettyMode(m)}}</button>`
    ).join('');
    container.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
            currentBrawlerMode = btn.dataset.mode;
            if (currentBrawlerMode !== 'all') currentBrawlerType = 'all';
            renderBrawlerModes();
            renderBrawlerTypes();
            renderBrawlerGrid();
        }});
    }});
}}

function renderBrawlerTypes() {{
    const container = document.getElementById('brawlerTypeFilter');
    const typeKeys = Object.keys(tierLabels);
    container.innerHTML = typeKeys.map(k =>
        `<button class="mode-btn ${{k === currentBrawlerType ? 'active' : ''}}" data-type="${{k}}">${{tierLabels[k]}}</button>`
    ).join('');
    container.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
            currentBrawlerType = btn.dataset.type;
            if (currentBrawlerType !== 'all') currentBrawlerMode = 'all';
            renderBrawlerModes();
            renderBrawlerTypes();
            renderBrawlerGrid();
        }});
    }});
}}

function enrichRow(r) {{
    // Ensure every row has raw_wr, wilson_lower, wilson_upper, adjusted_wr
    const wins = r.wins;
    const total = r.total;
    if (r.raw_wr === undefined) r.raw_wr = r.win_rate || (total ? +(100 * wins / total).toFixed(2) : 0);
    if (r.wilson_lower === undefined) {{
        const wl = wilsonLower(wins, total);
        r.wilson_lower = +(100 * wl).toFixed(2);
        r.wilson_upper = +(100 * Math.min(1, 2 * (wins/total || 0.5) - wl + 0.5/total)).toFixed(2);
        // Proper upper bound
        const z = 1.96, p = total ? wins/total : 0, z2 = z*z;
        const d = 1 + z2/total, c = (p + z2/(2*total))/d;
        const s = z * Math.sqrt((p*(1-p) + z2/(4*total))/total) / d;
        r.wilson_upper = +(100 * Math.min(1, c + s)).toFixed(2);
    }}
    if (r.adjusted_wr === undefined) r.adjusted_wr = r.raw_wr;
    return r;
}}

function renderBrawlerGrid() {{
    const useScores = (currentBrawlerMode === 'all' && currentBrawlerType === 'all');
    let rows;
    if (useScores) {{
        rows = (DATA.brawler_scores || []).map(r => ({{...r}}));
    }} else {{
        let key = currentBrawlerMode;
        if (currentBrawlerType !== 'all') key = currentBrawlerType;
        rows = (DATA.brawler_rates[key] || []).map(r => ({{...r}}));
    }}
    rows.forEach(enrichRow);
    rows.sort((a, b) => (b[currentSort] || 0) - (a[currentSort] || 0));

    const grid = document.getElementById('brawlerGrid');
    if (rows.length === 0) {{
        grid.innerHTML = '<div style="padding:20px;color:var(--text-dim)">No data for this filter combination.</div>';
        return;
    }}

    grid.innerHTML = rows.map(r => {{
        const url = portraitUrl(r.brawler_name);
        const imgTag = url ? `<img src="${{url}}" alt="${{r.brawler_name}}">` : `<div style="width:40px;height:40px;border-radius:50%;background:var(--border)"></div>`;
        const wr = r.raw_wr;
        const wins = r.wins;
        const total = r.total;

        // CI bar: scale 35%-65% range to 0-100% width
        const lo = Math.max(0, (r.wilson_lower - 35) / 30 * 100);
        const hi = Math.min(100, (r.wilson_upper - 35) / 30 * 100);
        const mid = Math.min(100, Math.max(0, (wr - 35) / 30 * 100));
        const ciHtml = `<div class="ci-bar"><div class="ci-fill" style="left:${{lo}}%;width:${{hi-lo}}%"></div><div class="ci-mark" style="left:${{mid}}%"></div></div>`;

        let metaHtml = `<div class="brawler-meta">adj ${{r.adjusted_wr}}% &middot; CI [${{r.wilson_lower}}%, ${{r.wilson_upper}}%]</div>`;

        // Per ranked tier mini-line (only when we have full score data)
        if (useScores && r.wr_Legendary !== undefined) {{
            const tierParts = rankedTierNames.map(t => {{
                const w = r['wr_' + t];
                const n = r['n_' + t];
                if (w == null || n < 3) return `<span style="opacity:0.3">${{t[0]}}: -</span>`;
                return `<span class="${{wrClass(w)}}">${{t[0]}}:${{Math.round(w)}}%</span>`;
            }}).join(' ');
            metaHtml += `<div class="brawler-meta" style="margin-top:1px">Ranked: ${{tierParts}}</div>`;
        }}

        return `<div class="brawler-card" style="flex-direction:column;align-items:stretch;padding:10px 12px">
            <div style="display:flex;align-items:center;gap:10px">
                ${{imgTag}}
                <div class="brawler-info">
                    <div class="brawler-name">${{r.brawler_name}}</div>
                    <div class="brawler-stats">${{wins}}W ${{total - wins}}L (${{total}} games)</div>
                </div>
                <div class="win-rate ${{wrClass(wr)}}">${{wr}}%</div>
            </div>
            ${{metaHtml}}${{ciHtml}}
        </div>`;
    }}).join('');
}}

renderSortToggle();
renderBrawlerModes();
renderBrawlerTypes();
renderBrawlerGrid();

// Combo win rates
let currentComboMode = 'all';
function renderComboModes() {{
    const modes = ['all', ...Object.keys(DATA.combos).filter(m => m !== 'all')];
    const container = document.getElementById('comboModeFilter');
    container.innerHTML = modes.map(m =>
        `<button class="mode-btn ${{m === currentComboMode ? 'active' : ''}}" data-mode="${{m}}">${{m === 'all' ? 'All Modes' : prettyMode(m)}}</button>`
    ).join('');
    container.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
            currentComboMode = btn.dataset.mode;
            renderComboModes();
            renderComboTable();
        }});
    }});
}}

function renderComboTable() {{
    const rows = DATA.combos[currentComboMode] || [];
    const tbody = document.querySelector('#comboTable tbody');
    tbody.innerHTML = rows.map(r => {{
        const brawlers = r.brawlers || r.combo.split(' + ');
        const portraits = brawlers.map(b =>
            portraitImg(b, 'portrait-sm') || `<span style="display:inline-block;width:24px;height:24px;border-radius:50%;background:var(--border)"></span>`
        ).join('<span class="combo-plus">+</span>');
        return `<tr>
            <td><div class="combo-portraits">${{portraits}}<span style="margin-left:6px;font-size:0.85em">${{brawlers.join(' + ')}}</span></div></td>
            <td><span class="win-rate ${{wrClass(r.win_rate)}}">${{r.win_rate}}%</span></td>
            <td>${{r.wins}}</td>
            <td>${{r.total}}</td>
        </tr>`;
    }}).join('');
}}

renderComboModes();
renderComboTable();

// Matchups
function renderMatchupTable(filter) {{
    filter = (filter || '').toUpperCase();
    const rows = DATA.matchups.filter(r =>
        !filter || r.brawler_a.includes(filter) || r.brawler_b.includes(filter)
    );
    const tbody = document.querySelector('#matchupTable tbody');
    tbody.innerHTML = rows.slice(0, 200).map(r => `<tr>
        <td>${{portraitImg(r.brawler_a)}}${{r.brawler_a}}</td>
        <td>${{portraitImg(r.brawler_b)}}${{r.brawler_b}}</td>
        <td><span class="win-rate ${{wrClass(r.win_rate)}}">${{r.win_rate}}%</span></td>
        <td>${{r.total}}</td>
    </tr>`).join('');
}}
renderMatchupTable();
document.getElementById('matchupSearch').addEventListener('input', e => renderMatchupTable(e.target.value));

// Synergies
function renderSynergyTable(filter) {{
    filter = (filter || '').toUpperCase();
    const rows = DATA.synergies.filter(r =>
        !filter || r.brawler_a.includes(filter) || r.brawler_b.includes(filter)
    );
    const tbody = document.querySelector('#synergyTable tbody');
    tbody.innerHTML = rows.slice(0, 200).map(r => `<tr>
        <td>${{portraitImg(r.brawler_a)}}${{r.brawler_a}}</td>
        <td>${{portraitImg(r.brawler_b)}}${{r.brawler_b}}</td>
        <td><span class="win-rate ${{wrClass(r.win_rate)}}">${{r.win_rate}}%</span></td>
        <td>${{r.total}}</td>
    </tr>`).join('');
}}
renderSynergyTable();
document.getElementById('synergySearch').addEventListener('input', e => renderSynergyTable(e.target.value));

// My Data tab
const MY = DATA.my_data;
let myLogMode = 'all';

function renderMyData() {{
    if (!MY) {{
        document.getElementById('myProfile').innerHTML = '<div style="padding:20px;color:var(--text-dim)">No personal data found. Make sure MAJOR_ACCOUNT_TAG is set in api.env and the battlelog has been fetched.</div>';
        return;
    }}

    // Profile card
    document.getElementById('myProfile').innerHTML = `
        <div class="stats-bar">
            <div class="stat"><div class="stat-val">${{MY.name}}</div><div class="stat-label">${{MY.tag}}</div></div>
            <div class="stat"><div class="stat-val">${{(MY.trophies || 0).toLocaleString()}}</div><div class="stat-label">Trophies</div></div>
            <div class="stat"><div class="stat-val">${{(MY.highest_trophies || MY.trophies || 0).toLocaleString()}}</div><div class="stat-label">Highest</div></div>
            <div class="stat"><div class="stat-val">${{MY.exp_level || '-'}}</div><div class="stat-label">Level</div></div>
            <div class="stat"><div class="stat-val">${{MY.club || '-'}}</div><div class="stat-label">Club</div></div>
            <div class="stat"><div class="stat-val">${{MY.battle_count}}</div><div class="stat-label">Tracked Battles</div></div>
        </div>`;

    // Brawler stats grid
    const grid = document.getElementById('myBrawlerGrid');
    grid.innerHTML = (MY.brawler_stats || []).map(r => {{
        const url = portraitUrl(r.brawler_name);
        const imgTag = url ? `<img src="${{url}}" alt="${{r.brawler_name}}">` : `<div style="width:40px;height:40px;border-radius:50%;background:var(--border)"></div>`;
        return `<div class="brawler-card">
            ${{imgTag}}
            <div class="brawler-info">
                <div class="brawler-name">${{r.brawler_name}}</div>
                <div class="brawler-stats">${{r.wins}}W ${{r.total - r.wins}}L (${{r.total}} games)${{r.star_count ? ' ⭐' + r.star_count : ''}}</div>
            </div>
            <div class="win-rate ${{wrClass(r.win_rate)}}">${{r.win_rate}}%</div>
        </div>`;
    }}).join('');

    // Mode stats
    const modeDiv = document.getElementById('myModeStats');
    modeDiv.innerHTML = '<div style="display:flex;gap:10px;flex-wrap:wrap">' +
        (MY.mode_stats || []).map(r => `
            <div class="brawler-card" style="min-width:140px">
                <div class="brawler-info">
                    <div class="brawler-name">${{prettyMode(r.mode)}}</div>
                    <div class="brawler-stats">${{r.wins}}W ${{r.total - r.wins}}L</div>
                </div>
                <div class="win-rate ${{wrClass(r.win_rate)}}">${{r.win_rate}}%</div>
            </div>`
        ).join('') + '</div>';

    document.getElementById('myBattleCount').textContent = `(${{MY.battle_count}} tracked)`;
    renderMyLogFilter();
    renderMyLog();
}}

function renderMyLogFilter() {{
    if (!MY) return;
    const modes = ['all', ...new Set(MY.battle_log.map(b => b.mode))];
    const container = document.getElementById('myLogFilter');
    container.innerHTML = modes.map(m =>
        `<button class="mode-btn ${{m === myLogMode ? 'active' : ''}}" data-mode="${{m}}">${{m === 'all' ? 'All' : prettyMode(m)}}</button>`
    ).join('');
    container.querySelectorAll('.mode-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
            myLogMode = btn.dataset.mode;
            renderMyLogFilter();
            renderMyLog();
        }});
    }});
}}

function renderMyLog() {{
    if (!MY) return;
    let log = MY.battle_log;
    if (myLogMode !== 'all') log = log.filter(b => b.mode === myLogMode);

    const tbody = document.querySelector('#myLogTable tbody');
    tbody.innerHTML = log.map(b => {{
        const time = b.time ? b.time.slice(0, 16).replace('T', ' ') : '?';
        const resultCls = b.result === 'victory' ? 'wr-high' : b.result === 'defeat' ? 'wr-low' : 'wr-mid';
        const star = b.star_player ? ' ⭐' : '';
        const dur = b.duration ? ` (${{b.duration}}s)` : '';
        const tc = b.trophy_change ? ` ${{b.trophy_change > 0 ? '+' : ''}}${{b.trophy_change}}` : '';

        const tmStr = (b.teammates || []).map(t =>
            `${{portraitImg(t.brawler)}}${{t.brawler}}`
        ).join(', ') || '-';
        const opStr = (b.opponents || []).map(o =>
            `${{portraitImg(o.brawler)}}${{o.brawler}}`
        ).join(', ') || '-';

        return `<tr>
            <td style="white-space:nowrap;font-size:0.85em">${{time}}</td>
            <td>${{prettyMode(b.mode)}}</td>
            <td style="font-size:0.85em">${{b.map || '-'}}</td>
            <td>${{portraitImg(b.brawler)}}${{b.brawler}}</td>
            <td><span class="${{resultCls}}" style="font-weight:600">${{b.result}}${{star}}${{tc}}</span>${{dur}}</td>
            <td style="font-size:0.85em">${{tmStr}}</td>
            <td style="font-size:0.85em">${{opStr}}</td>
        </tr>`;
    }}).join('');
}}

renderMyData();
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    html_content = ""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.html_content.encode())

    def log_message(self, format, *args):
        pass


def fetch_remote_cache(ssh_host: str, remote_path: str = "brawlstar-agent/data/analytics_cache.json") -> bool:
    """rsync the analytics cache from a remote host into the local CACHE_PATH.

    Returns True on success, False on any rsync error or timeout. The local
    cache file is preserved on failure so the dashboard can still launch with
    whatever's there.
    """
    if not shutil.which("rsync"):
        print("WARN: rsync not installed locally; skipping remote cache fetch", file=sys.stderr)
        return False

    remote = f"{ssh_host}:{remote_path}"
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching analytics cache from {ssh_host}...", flush=True)
    try:
        result = subprocess.run(
            ["rsync", "-az", "--timeout=30", remote, str(CACHE_PATH)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("WARN: rsync timed out after 60s; using whatever local cache exists.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"WARN: rsync failed: {exc}", file=sys.stderr)
        return False

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        print(f"WARN: rsync exited {result.returncode}; falling back to local cache.\n{stderr}", file=sys.stderr)
        return False

    return True


def _format_cache_age(computed_at_iso: str) -> str:
    """Render '23 minutes ago' / '4 hours ago' for the cache header."""
    try:
        computed = datetime.fromisoformat(computed_at_iso)
    except (TypeError, ValueError):
        return "unknown age"
    now = datetime.now(timezone.utc)
    if computed.tzinfo is None:
        computed = computed.replace(tzinfo=timezone.utc)
    delta = now - computed
    secs = delta.total_seconds()
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs / 60)} min ago"
    if secs < 86400:
        return f"{secs / 3600:.1f} h ago"
    return f"{secs / 86400:.1f} d ago"


def main():
    parser = argparse.ArgumentParser(description="Brawl Stars analytics dashboard")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute, ignoring cache (also writes the cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache entirely; compute inline without writing.",
    )
    parser.add_argument(
        "--remote-cache",
        metavar="SSH_HOST",
        help="Before launching, rsync the analytics cache from a remote SSH host "
             "(e.g., 'brawl'). Falls back to local cache on failure.",
    )
    args = parser.parse_args()

    if args.remote_cache:
        fetch_remote_cache(args.remote_cache)

    cache_meta: dict | None = None
    if args.recompute:
        print("Forcing recompute (--recompute), this may take a while...")
        cache = write_cache(args.db)
        data = cache["data"]
        cache_meta = {k: v for k, v in cache.items() if k != "data"}
    elif args.no_cache:
        print("Computing analytics inline (--no-cache)...")
        data = collect_all_data(args.db)
    else:
        cache = read_cache()
        if cache and "data" in cache:
            age = _format_cache_age(cache.get("computed_at", ""))
            print(f"Using cached analytics from {age} (cache: {CACHE_PATH})")
            data = cache["data"]
            cache_meta = {k: v for k, v in cache.items() if k != "data"}
        else:
            print("No cache found; computing inline (this is slow on the droplet)...")
            print("Tip: run scripts/precompute-analytics.py periodically, or pass --recompute.")
            data = collect_all_data(args.db)

    # Embed cache metadata so the HTML can render a freshness banner.
    data["_cache_meta"] = cache_meta

    print("Loading brawler portraits...")
    portraits = load_portrait_map()
    name_to_id = load_brawler_name_map()
    print(f"  {len(portraits)} portraits loaded")

    print("Generating dashboard HTML...")
    html = generate_html(data, portraits, name_to_id)

    DashboardHandler.html_content = html
    server = HTTPServer(("localhost", args.port), DashboardHandler)

    url = f"http://localhost:{args.port}"
    print(f"\nDashboard ready at {url}")
    print("Press Ctrl+C to stop\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

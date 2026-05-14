# Technical Context

## Brawl Stars game-domain semantics (CRITICAL — read before touching the recommender)

The two competitive `battle_type` values in our DB do **not** match their in-game-UI names. Internalize this:

| DB `battle_type` | In-game name | Has draft? | What it actually is |
|---|---|---|---|
| `'ranked'` | **Unranked** (trophy battles) | No | Regular trophy ladder. No tier system, no ban-pick. Players queue with whatever brawler. |
| `'soloRanked'` | **Ranked** | Tier-dependent (see below) | The competitive ranked queue with Bronze → Silver → Gold → Diamond → Mythic → Legendary → Masters → Pro tiers. |

**Draft mechanism by tier** in the `'soloRanked'` queue:

| Tier | DB `brawler_trophies` value | Draft mechanism |
|---|---:|---|
| Bronze | 1–3 | None / open pick |
| Silver | 4–6 | None / open pick |
| Gold | 7–9 | None / open pick |
| Diamond | 10–12 | Add ban stage, but still simultaneous pick (lax — both teams choose at once, less strategic) |
| **Mythic** | **13–15** | **Strict 1-2-2-1 ban-pick draft** ← serious competitive starts here |
| **Legendary** | **16–18** | **Strict 1-2-2-1 ban-pick draft** |
| **Masters** | **19** | **Strict 1-2-2-1 ban-pick draft** |
| Pro | 20–22 | Strict 1-2-2-1 ban-pick draft |

So when we say "competitive draft games" we specifically mean `battle_type = 'soloRanked' AND brawler_trophies >= 13` (Mythic+). Diamond (10–12) is excluded because simultaneous pick is laxer than the strict 1-2-2-1 order.

**`brawler_trophies` field is overloaded** — same column, two different scales (verified 2026-05-09):
- In `'ranked'` battles: cumulative brawler trophy count, range 0–4951, mean ≈ 1040, var ≈ 356 k.
- In `'soloRanked'` battles: tier number 1–22, range 1–22, mean ≈ 14.07, var ≈ 10.

This means features built off `brawler_trophies` (and any aggregates like `team_a_trophies_mean`, `team_a_trophies_min/max/std`) carry mixed semantics if computed across both `battle_type` values without conditioning. The phase-1 ablation diagnosis (DEC-015 follow-up) showed this contributing to the `P1.big` regression — modes with higher `soloRanked` content tracked closer to "no improvement", and the largest gains were in `soloRanked`-free modes (`siege`, `basketBrawl`).

**3v3 mode whitelist** (clean ranked / soloRanked window, post-2026-05-03):
brawlBall, knockout, gemGrab, bounty, hotZone, heist, siege, basketBrawl, wipeout — 9 modes, all 3v3.
`duels` (1v1) also appears in the same `battle_type` filter but `dataset.py:require_complete_teams=True` implicitly drops it. Will be made explicit when we touch `dataset.py` next.

## Host Machine

| Property | Value |
|----------|-------|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-52-generic |
| CPU | Intel i7-12700H (14C/20T) |
| RAM | 62 GB |
| GPU | NVIDIA RTX 3060 Mobile, driver 535.230.02, CUDA 12.2 |
| Disk | /media/lin/disk2 — 456GB NVMe, project lives here |
| Home disk | /home — only 27GB free, avoid writing here |

## Software

| Tool | Version |
|------|---------|
| Python (uv venv) | 3.12.13 |
| uv | 0.10.9 |
| OpenCV | 4.13.0 |
| numpy | 2.4.3 (downgraded from 2.4.4 by torch CPU index pinning, no functional impact) |
| scikit-learn | 1.8.0 |
| LightGBM | 4.6.0 |
| pandas | 3.0.2 |
| **PyTorch** | **2.5.1+cu121** (added Session 9 morning as 2.11.0+cpu; swapped same day to cu121 once `nvidia-modprobe` was installed; pinned `>=2.5.0,<2.6` because cu121 wheels stopped at 2.5.1 — PyTorch ≥2.6 is cu124-only and our driver 535 doesn't support cu124) |
| nvidia-cudnn-cu12 | 9.1.0 (pulled in by torch+cu121) |
| triton | 3.1.0 (pulled in by torch+cu121) |
| yt-dlp | latest |
| tesseract | installed (apt) |
| ffmpeg | 4.4.2 |
| Docker | 28.0.1 |

**GPU note** (RTX 3060 Mobile, ENABLED in Session 9 evening):
- `lspci`: NVIDIA Corporation GA106M (GeForce RTX 3060 Mobile / Max-Q). Driver 535.230.02 (kernel modules nvidia, nvidia_uvm, nvidia_drm, nvidia_modeset all loaded). CUDA 12.2 toolkit at `/usr/local/cuda-12.2/`. **5.77 GB VRAM, compute capability 8.6** (Ampere).
- One-time enablement (DEC-013): `sudo apt install nvidia-modprobe`. Without it `/dev/nvidia*` device nodes never get created and torch + nvidia-smi both fail. Important: install must be done in a **real terminal**, not from inside Cursor's user-namespace sandbox — dpkg in the namespace writes the binary as `nobody:nogroup` (UID 65534), and the setuid bits then drop privileges to nobody, which can't write to /dev. If accidentally done in Cursor: `sudo chown root:root /usr/bin/nvidia-modprobe` to fix.
- After `nvidia-modprobe`, the CPU torch wheel was swapped for cu121 via `uv add torch --index https://download.pytorch.org/whl/cu121`. `pyproject.toml` routes torch through a named explicit index (`pytorch-cu121`) so opencv / jupyter / etc. continue resolving from PyPI. A `pytorch-cpu` named index is also defined for easy revert (e.g. droplet builds).
- v3 transformer training on GPU (RTX 3060 Mobile, fast data path):
  - Small arch (251 k params, batch 4096): ~40 s / epoch → 5.6 min for 6 epochs
  - Big arch (570 k params, batch 4096): ~95 s / epoch → 14 min for 8 epochs
  - vs CPU same arch: 4.9× speedup end-to-end (data plumbing + GPU compute combined).

## Production Droplet (DigitalOcean)

| Property | Value |
|----------|-------|
| Provider / plan | DigitalOcean Basic, $6/mo, US region |
| OS | Ubuntu 24.04 LTS |
| Resources | 1 vCPU / 1 GB RAM / 25 GB SSD |
| Reserved IPv4 | `209.38.4.212` (inbound only — SSH lands here) |
| Anchor IPv4 | `64.23.171.86` (outbound source IP — what BS API sees) |
| User | `lin` (sudo), key-only SSH; root login disabled |
| **Login shell** | **`/bin/bash`** (unchanged; `chsh` NOT used) |
| **Interactive shell** | fish (auto-exec'd from `~/.bashrc` for TTY sessions only — `NO_FISH=1` to skip) |
| **Multiplexer** | tmux + zellij both installed at `/usr/local/bin/` |
| Code path | `/home/lin/brawlstar-agent/` |
| DB path | `/home/lin/brawlstar-agent/data/brawlstars.db` |
| Cron | `brawl-collect.timer` every 6h → `brawl-collect.service` |
| Backup | (planned) Cloudflare R2 nightly via `rclone` |

**Shell-layering note**: `~/.bashrc` runs first (sources `UV_CACHE_DIR`, `BRAWL_API_KEY_VAR`), then auto-execs fish only if `-t 1 && -z $NO_FISH && -z $INSIDE_FISH`. Non-interactive `ssh brawl 'cmd'` stays in bash. systemd units never see fish (absolute paths + `Environment=` directives). See `docs/deployment.md` § 16 for the full pattern + gotcha table.

**Ops convention — droplet commands the agent suggests to the user**: present them as **bare commands** the user pastes after `ssh brawl`. Do NOT wrap them as `ssh brawl 'cmd'` in chat — the user can't paste that into an already-open droplet session and has to mentally peel off the wrapper. Two reasons compound:

1. `sudo` on the droplet prompts for a password; non-interactive `ssh "$host" "sudo ..."` can't accept it cleanly.
2. The user often has an interactive `ssh brawl` session open already and just wants commands they can paste verbatim.

Convention captured in `CLAUDE.md` § "Communicating commands to the user". The agent's own Shell-tool calls may still use `ssh brawl 'cmd'` for non-interactive read-only checks — that path is invisible to the user.

**Ops convention — rsync paths to/from the droplet**: rsync ≥ 3.2.4 enables `--secluded-args` by default. Path arguments after `host:` are sent straight to the remote `rsync` and are **not interpreted by the remote shell**, so `$HOME` (and other shell variables) stay literal. Use a **relative path** (`brawlstar-agent/data/brawlstars.db`) or a **tilde** (`~/brawlstar-agent/data/brawlstars.db`) on the remote side — both are handled by rsync itself even with secluded-args, and both also work in the `ssh "$host" "sqlite3 $REMOTE_PATH ..."` companion form (relative paths resolve against the non-interactive ssh CWD = `$HOME`). Symptom of getting it wrong: `rsync: [sender] change_dir "/home/<user>/$HOME/..." failed: No such file or directory (2)`. Both `scripts/rsync-db-{from,to}-droplet.sh` were fixed on 2026-05-13 to use relative paths; the gotcha is also captured in `docs/deployment.md` § 16.

## Brawl Stars API

| Property | Value |
|----------|-------|
| Base URL | `https://api.brawlstars.com/v1` |
| Auth | Bearer token (JWT, IP-locked, developer/silver tier) |
| Credentials | `api.env` (git-ignored): `BRAWL_STAR_API` (local IP), `BRAWL_STAR_API_DO` (droplet IPs), `MAJOR_ACCOUNT_TAG=#RYY9LJVL` |
| Per-machine selection | `BRAWL_API_KEY_VAR` env var (set in `~/.bashrc` and systemd unit) — defaults to `BRAWL_STAR_API` |
| Docs | `docs/brawlstars-api.md` (full reference) |
| Example data | `docs/api-examples/` (live responses from 2026-04-13) |
| Brawler count | 104 as of 2026-05-04 (was 101 in last collection; Supercell added 3) |
| Rate limit | Undocumented, throttles aggressive scanning; we run at 2 req/s with backoff |

Endpoints in use: player profile, battlelog, global rankings, brawler list, brawler detail, game modes, event rotation.

### Battle types — API field `battle.type` vs in-game UI (IMPORTANT, the names are flipped)

The API names look misleading. Always read them literally per this table, not by what they sound like:

| API `battle.type` | In-game tab | What it actually is |
|---|---|---|
| `ranked` | **Trophy Battles** (casual / "unranked") | Trophy ladder. Free pick, no bans. `trophyChange` present (`brawler.trophies` is the real per-brawler trophy count). Covers every mode incl. Showdown (solo + duo) and 3v3. **The everyday casual mode** — confusing API name. |
| `soloRanked` | **Ranked** (the actual competitive ladder) | Tier progression: **Bronze → Silver → Gold → Diamond → Mythic → Legendary → Masters → Pro**. **Has ban-pick draft after a certain rank tier** (Diamond+ per common knowledge — exact threshold not measured here). No `trophyChange`; `brawler.trophies` is the rank tier index (1‑22), **not** trophies. Locked to 6-mode rotation: gemGrab, brawlBall, knockout, bounty, hotZone, heist. |
| `friendly` | Friendly room / club friendly | brawler.trophies = −1 sentinel, no trophyChange. Different player population, usually excluded from meta analysis. |
| `challenge` | Championship / Power League / event challenges | `brawler.trophies` 0‑803 is a challenge-internal progression counter, `trophyChange = 1` is a "you played 1 challenge match" counter. |
| `tournament` | In-game player-organized tournaments | Tiny volume. brawler.trophies = −1. |
| (NULL) | PvE special events (`lastStand` on "BOSS CROW") | API omits `type` for these. Filter out for any 3v3 analytics. |

**Recommender-relevant implication of ban-pick in soloRanked**: the API only returns the 6 *picked* brawlers per battle, not the bans. The recommender therefore conditions on the post-ban candidate set without knowing which brawlers were banned — a structurally missing feature for soloRanked rows. Worth keeping in mind when interpreting per-mode AUC differences between `ranked` (no bans) and `soloRanked` (post-ban) data, and as a future feature-engineering candidate if/when ban data ever becomes available.

Recommender filter is `COMPETITIVE_BATTLE_TYPES = ('ranked', 'soloRanked')` in `src/brawlstar_agent/recommender/dataset.py`. Both contribute to training; soloRanked is ~10‑11 % of clean rows, ranked is ~85‑89 %.

## Emulator Software (installed but not useful for Brawl Stars)

| Tool | Status |
|------|--------|
| Android SDK | cmdline-tools 12.0, platform-tools 37.0.0, emulator 36.4.10 |
| Genymotion | 3.9.0 (no ARM translation) |
| JDK | Temurin 17.0.18+8 (on disk2) |

## Cursor IDE Sandbox (Ubuntu 22.04 quirk)

The `cursor_sandbox` AppArmor profile shipped by the Cursor `.deb` (≥ 3.2.16, file `/etc/apparmor.d/cursor-sandbox`) is incomplete on AppArmor 3.0.4. It omits `capability dac_override,`, which `newuidmap` needs to write `/proc/<pid>/uid_map`. With the profile loaded, agent autorun preflight fails:

- Symptom: extHost log shows `Sandbox support detected: false`; auto-run stays unavailable.
- Audit: `journalctl | grep apparmor=\"DENIED\"` shows `profile="cursor_sandbox" comm="newuidmap" capname="dac_override"`.

Fix already applied (2026-05-02): profile disabled via the standard symlink mechanism.

```bash
sudo ln -sf /etc/apparmor.d/cursor-sandbox /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/cursor-sandbox
```

Persists across reboots — the apparmor service skips profiles symlinked into `disable/`. Safe because `kernel.apparmor_restrict_unprivileged_userns=0` already permits unconfined userns creation, and Cursor's actual sandbox (Landlock + seccomp + namespaces) runs *inside* `cursorsandbox` regardless of the outer AppArmor.

Caveat: a Cursor `.deb` upgrade may run `apparmor_parser -r` in postinst and re-load the profile until next reboot. If autorun breaks after an upgrade, re-run the `apparmor_parser -R` line. The `disable/` symlink itself survives upgrades.

Re-enable the profile if/when host moves to Ubuntu 24.04+ (AppArmor 4.0): the profile's commented-out `userns,` directive then becomes meaningful and the profile would be the proper "permanent fix" Cursor intends.

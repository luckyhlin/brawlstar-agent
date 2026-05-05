# Deployment Runbook — Brawl Stars Crawler on a Fresh VPS

This document is a step-by-step procedure for bringing the crawler up on a new VPS (e.g., a fresh DigitalOcean droplet, or a migration to another provider). It captures everything learned during the initial DO deploy on **2026-05-03** so future-us doesn't have to rediscover the gotchas.

> **Architecture context**: see [`memory-bank/decisions.md`](../memory-bank/decisions.md) DEC-007 (hosting choice) and DEC-008 (local-primary git workflow). This runbook is the *how*; those decisions are the *why*.

## Prerequisites

- A VPS account (DigitalOcean assumed below; the steps generalize to any Ubuntu 24.04 VPS with a static IPv4)
- A Brawl Stars developer account at [developer.brawlstars.com](https://developer.brawlstars.com) — you'll need to whitelist new IPs against an API key
- GitHub access to the project repo (for the `git clone` deploy step)
- An SSH key on your local machine (`~/.ssh/id_ed25519.pub` or similar)
- Optional: a Cloudflare account if you want R2 backups

## What you'll end up with

```
VPS (1 GB RAM, 25 GB SSD, $6/mo)
├── Ubuntu 24.04, hardened (fail2ban, UFW, no root SSH)
├── Python 3.12 + uv-managed venv
├── git-cloned repo at /home/<user>/brawlstar-agent
├── SQLite DB at .../data/brawlstars.db
└── systemd timer "brawl-collect.timer" → runs the crawler every 6h
```

---

## 1. Create the droplet

DigitalOcean web UI → Create → Droplet:

| Setting | Value |
|---|---|
| Region | Closest US region (NYC1/NYC3/SFO3/TOR1) |
| Image | Ubuntu **24.04 LTS x64** |
| Plan | Basic → Regular (SSD) → **$6/mo** (1 GB / 25 GB / 1 TB) |
| Authentication | **SSH Key** — upload `~/.ssh/id_ed25519.pub` from local |
| Hostname | `brawl-data` (or anything memorable) |
| IPv6 | Enable (free, harmless) |
| Backups / Monitoring / Block Storage | Skip — we'll handle backups via R2 |

After creation, **reserve an IP** and attach it to the droplet:

- Networking → Reserved IPs → Reserve → assign to droplet
- Free while attached to a droplet
- This is the IP that survives droplet rebuilds — useful for a stable inbound endpoint

> ⚠ **Critical gotcha**: DO Reserved IPs are **inbound-only** by default. Outbound traffic from the droplet still uses the original (anchor) public IP. The Brawl Stars API will see the *anchor* IP when your crawler makes requests, **not** the reserved IP.
>
> Verify the actual outbound IP with:
> ```bash
> ssh <user>@<reserved_ip>
> curl -s https://api.ipify.org
> ```
> Whitelist **both** IPs on the BS API key for safety.

## 2. Whitelist droplet IPs on the Brawl Stars API key

1. [developer.brawlstars.com](https://developer.brawlstars.com) → My Account → your key (or create a new one)
2. Add **both** the reserved IP and the anchor public IP to "Allowed IP addresses"
3. Save and copy the new JWT key
4. Store it in `api.env` (locally) under a distinct name like `BRAWL_STAR_API_DO`

## 3. First SSH and create non-root user

From your laptop:

```bash
ssh root@<reserved_ip>
```

On the droplet:

```bash
adduser lin                    # set a strong password — needed for sudo
usermod -aG sudo lin

# Copy SSH key to new user
mkdir -p /home/lin/.ssh
cp ~/.ssh/authorized_keys /home/lin/.ssh/authorized_keys
chown -R lin:lin /home/lin/.ssh
chmod 700 /home/lin/.ssh
chmod 600 /home/lin/.ssh/authorized_keys
```

**Test from a SECOND terminal on your laptop** (don't close the root session yet):

```bash
ssh lin@<reserved_ip>
sudo whoami    # should print: root
```

## 4. UFW firewall

In the still-open root session:

```bash
ufw allow OpenSSH
ufw default deny incoming
ufw default allow outgoing
ufw enable    # type 'y' when warned about SSH
ufw status verbose
```

Verify the `lin` SSH session still works after enabling.

## 5. Lock down SSH

Edit `/etc/ssh/sshd_config`:

```
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

Validate and restart:

```bash
sshd -t                      # no output = no syntax errors
systemctl restart ssh
```

**Test from a third terminal**:

```bash
ssh lin@<reserved_ip>        # should still work
ssh root@<reserved_ip>       # should be REJECTED (this is the goal)
```

## 6. Install fail2ban with home-IP allowlist

On the droplet (as `lin`):

```bash
sudo apt update
sudo apt install -y fail2ban

# Get your home IPv4 from your LAPTOP (not the droplet):
#   curl -s -4 ifconfig.me
# Substitute it below.
MY_HOME_IP="76.102.167.155"

sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
ignoreip = 127.0.0.1/8 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 ${MY_HOME_IP}
maxretry = 8
findtime = 600
bantime  = 3600

[sshd]
enabled = true
EOF

sudo systemctl restart fail2ban
sleep 2     # avoid race with socket init
sudo fail2ban-client status sshd
```

> ⚠ **Pitfall**: `fail2ban-client status` immediately after `systemctl restart` may fail with `Failed to access socket path`. The daemon takes ~1 second to bind its socket. `sleep 2` between commands fixes it. Default `bantime=10m` is also too short; 1h is more useful.

## 7. Defuse needrestart

Without this, every future `apt install` or `apt upgrade` will silently restart your crawler mid-run.

```bash
sudo sed -i "s/^#\$nrconf{restart} = .*/\$nrconf{restart} = 'l';/" /etc/needrestart/needrestart.conf
grep '^\$nrconf{restart}' /etc/needrestart/needrestart.conf
# expected output: $nrconf{restart} = 'l';
```

The `'l'` means "list services that need restart, but don't actually restart them." We'll restart manually when *we* want them.

## 8. Install Python toolchain + sqlite3 CLI

```bash
sudo apt install -y git python3 python3-venv python3-dev build-essential rsync sqlite3
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
which uv          # should be /home/lin/.local/bin/uv
python3 --version # should be 3.12.x
```

## 9. Set up GitHub deploy key

The droplet pulls code via `git pull`. Generate a new SSH key on the droplet (do **not** copy your laptop's key):

```bash
ssh-keygen -t ed25519 -C "brawl-data-droplet" -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the output. In GitHub:

- Repo → Settings → Deploy keys → Add deploy key
- Title: `brawl-data droplet (read-only)`
- Paste the key
- **Leave "Allow write access" unchecked** (read-only is enough; we never push from droplet)

Test:

```bash
ssh -T git@github.com
# expected: "Hi <repo>! You've successfully authenticated, but GitHub does not provide shell access."
```

## 10. Clone the repo and configure env

```bash
cd ~
git clone git@github.com:<owner>/brawlstar-agent.git
cd brawlstar-agent

# Per-machine env vars
echo 'export UV_CACHE_DIR=$HOME/.cache/uv' >> ~/.bashrc
echo 'export BRAWL_API_KEY_VAR=BRAWL_STAR_API_DO' >> ~/.bashrc
source ~/.bashrc
```

> The `BRAWL_API_KEY_VAR` indirection is what makes `api_client.py` work with different IP-locked keys per machine. See `src/brawlstar_agent/api_client.py::_load_key`.

## 11. Restore gitignored files

These don't live in git. Transfer from local:

```bash
# From your LAPTOP:
rsync -avz /media/lin/disk2/brawlstar-agent/api.env \
  lin@<reserved_ip>:/home/lin/brawlstar-agent/api.env

rsync -avz --progress /media/lin/disk2/brawlstar-agent/data/brawlstars.db \
  lin@<reserved_ip>:/home/lin/brawlstar-agent/data/brawlstars.db
```

Or if you want a fresh DB on the droplet (no historical local data), just create the directory; the schema auto-creates on first run:

```bash
# On droplet:
mkdir -p ~/brawlstar-agent/data
```

## 12. Install dependencies

```bash
cd ~/brawlstar-agent
uv sync
```

Should take 1-3 minutes. If you get a permission error about `/media/lin/disk2/.uv-cache`, check that you've removed `[tool.uv] cache-dir` from `pyproject.toml` (we did this on 2026-05-03) and that `UV_CACHE_DIR` is set in `~/.bashrc`.

## 13. End-to-end sanity test

```bash
cd ~/brawlstar-agent
PYTHONPATH=src uv run python -c "
from brawlstar_agent.api_client import BrawlStarsAPI
api = BrawlStarsAPI()
brawlers = api.get_brawlers()
print(f'OK: fetched {len(brawlers)} brawlers, first: {brawlers[0][\"name\"]}')
api.close()
"
```

Expected: `OK: fetched 104 brawlers, first: SHELLY` (count varies as Supercell ships new brawlers).

If you get **403 accessDenied**: the IP whitelist on the API key doesn't match the droplet's actual outbound IP. Re-verify with `curl -s https://api.ipify.org` and update the BS API key.

## 14. systemd services + timers (3 timers in total)

The droplet runs three independent timers:

| Timer | Cadence | Service script | Purpose |
|---|---|---|---|
| `brawl-collect.timer` | every 6h | `scripts/collect-battles.py --collect-only ...` | Bulk snowball; tops up battlelogs for high-trophy players |
| `brawl-collect-pinned.timer` | every 1h | `scripts/collect-pinned.py` | Always-fetch a small list of personal/inspection tags from `data/pinned_tags.txt` |
| `brawl-analytics.timer` | every 1h | `scripts/precompute-analytics.py` | Pre-compute heavy SQL into `data/analytics_cache.json` so the dashboard launches instantly |

### 14a. Bulk crawler (every 6h)

```bash
sudo tee /etc/systemd/system/brawl-collect.service > /dev/null <<'EOF'
[Unit]
Description=Brawl Stars battle data collector
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=lin
Group=lin
WorkingDirectory=/home/lin/brawlstar-agent
Environment="BRAWL_API_KEY_VAR=BRAWL_STAR_API_DO"
Environment="UV_CACHE_DIR=/home/lin/.cache/uv"
Environment="PYTHONPATH=src"
ExecStart=/home/lin/.local/bin/uv run python scripts/collect-battles.py --collect-only --battlelog-limit 1500 --older-than 5.5 --rps 2
StandardOutput=journal
StandardError=journal
Nice=10
EOF

sudo tee /etc/systemd/system/brawl-collect.timer > /dev/null <<'EOF'
[Unit]
Description=Run Brawl Stars collector every 6 hours

[Timer]
OnBootSec=10min
OnUnitActiveSec=6h
RandomizedDelaySec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now brawl-collect.timer
systemctl list-timers brawl-collect.timer --no-pager
```

> ⚠ **Pitfall**: `Type=oneshot` services block `systemctl start` until the run completes (~15 min). Always use `--no-block` for manual triggers, OR start from a separate shell session. Pressing Ctrl+C on systemctl detaches you but does NOT kill the underlying service.

Manual trigger (from a separate shell so it doesn't block your prompt):

```bash
sudo systemctl start --no-block brawl-collect.service
journalctl -u brawl-collect.service -f    # Ctrl+C to detach
```

### 14b. Pinned-tag crawler (every 1h)

The bulk crawler ranks by trophies and only fetches the top ~1500 stale tags per run. With 500k+ tags discovered, your personal account (and other low-trophy tags you care about) effectively never get crawled. This timer always crawls a small explicit list.

First create the tags file. **Don't commit this file** — `data/` is already gitignored.

```bash
mkdir -p ~/brawlstar-agent/data
cat > ~/brawlstar-agent/data/pinned_tags.txt <<'EOF'
# Personal account
#RYY9LJVL

# Friends / players under inspection (inline # comments allowed):
# #280YJ0R80   # PolyMentos
# #2GY9CCUQR0  # psyduck
EOF
```

> **Tags vs. watched-player display**: this same file feeds two consumers — the pinned-tags crawler (`scripts/collect-pinned.py`) and the dashboard's "Watched Players" tab (`dashboard_data.py::collect_all_data`). The `MAJOR_ACCOUNT_TAG` (from `api.env`) gets the dedicated "My Data" tab; every *other* tag in this file shows up under "Watched Players" with a per-player subtab. So adding a friend's tag here gives you both periodic crawling AND a view in the dashboard with one edit.

Then the systemd unit pair:

```bash
sudo tee /etc/systemd/system/brawl-collect-pinned.service > /dev/null <<'EOF'
[Unit]
Description=Brawl Stars pinned-tag crawler (personal + watchlist)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=lin
Group=lin
WorkingDirectory=/home/lin/brawlstar-agent
Environment="BRAWL_API_KEY_VAR=BRAWL_STAR_API_DO"
Environment="UV_CACHE_DIR=/home/lin/.cache/uv"
Environment="PYTHONPATH=src"
ExecStart=/home/lin/.local/bin/uv run python scripts/collect-pinned.py
StandardOutput=journal
StandardError=journal
Nice=10
EOF

sudo tee /etc/systemd/system/brawl-collect-pinned.timer > /dev/null <<'EOF'
[Unit]
Description=Run Brawl Stars pinned-tag crawler every 1 hour

[Timer]
OnBootSec=5min
OnUnitActiveSec=1h
RandomizedDelaySec=2min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now brawl-collect-pinned.timer
systemctl list-timers brawl-collect-pinned.timer --no-pager
```

### 14c. Analytics precompute (every 1h)

The dashboard's matchup/synergy queries do self-joins on ~1M `battle_players` rows — 5-15 min on a 1-CPU droplet. Pre-computing on a schedule and caching to JSON makes the dashboard load in <1 sec.

```bash
sudo tee /etc/systemd/system/brawl-analytics.service > /dev/null <<'EOF'
[Unit]
Description=Brawl Stars analytics precompute (writes data/analytics_cache.json)
After=network-online.target

[Service]
Type=oneshot
User=lin
Group=lin
WorkingDirectory=/home/lin/brawlstar-agent
Environment="UV_CACHE_DIR=/home/lin/.cache/uv"
Environment="PYTHONPATH=src"
ExecStart=/home/lin/.local/bin/uv run python scripts/precompute-analytics.py
# Hard kill if compute exceeds 45 min — indicates DB grew or query plan regressed.
# Failed unit will appear in `systemctl --failed`.
TimeoutStartSec=2700
StandardOutput=journal
StandardError=journal
Nice=15
EOF

sudo tee /etc/systemd/system/brawl-analytics.timer > /dev/null <<'EOF'
[Unit]
Description=Run Brawl Stars analytics precompute every 1 hour

[Timer]
OnBootSec=15min
OnUnitActiveSec=1h
RandomizedDelaySec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now brawl-analytics.timer
systemctl list-timers brawl-analytics.timer --no-pager
```

> The `TimeoutStartSec=2700` is the watchdog: if compute ever exceeds 45 min, systemd kills it and marks the service as failed. The dashboard also paints the cache header **orange** at >30 min and **red** at >45 min, so a slow regression is visible without needing to check logs.

Trigger the first analytics compute manually (so the dashboard has cache to read immediately):

```bash
sudo systemctl start --no-block brawl-analytics.service
journalctl -u brawl-analytics.service -f
```

When `Wrote cache to ... in NNs total` appears in the journal, `data/analytics_cache.json` is ready.

## 15. Verify data is landing

```bash
sqlite3 ~/brawlstar-agent/data/brawlstars.db \
  "SELECT COUNT(*), MAX(battle_time_iso) FROM battles;"
```

Numbers should tick up over time as scheduled runs accumulate.

---

## Day-2 operations

### Deploy code changes from local

```bash
# On your LOCAL machine
cd /media/lin/disk2/brawlstar-agent
git add . && git commit -m "..." && git push

# On droplet
ssh lin@<reserved_ip>
cd ~/brawlstar-agent
git pull
sudo systemctl restart brawl-collect.service   # only if a running collection should pick up the new code
```

### Inspect status

```bash
# All three timers at once
systemctl list-timers 'brawl-*' --no-pager

# Per-service status
systemctl status brawl-collect.service --no-pager
systemctl status brawl-collect-pinned.service --no-pager
systemctl status brawl-analytics.service --no-pager

# Recent logs
journalctl -u brawl-collect.service -n 50 --no-pager
journalctl -u brawl-collect-pinned.service -n 30 --no-pager
journalctl -u brawl-analytics.service -n 30 --no-pager
journalctl -u brawl-collect.service -f             # follow live

# Anything failing?
systemctl --failed
sudo fail2ban-client status sshd
```

### Tune cadence

Edit `/etc/systemd/system/brawl-collect.service` to change `--battlelog-limit` or `--rps`, then:

```bash
sudo systemctl daemon-reload
```

(No restart needed; next timer fire picks up the new args.)

### View the dashboard

**Recommended: run the dashboard locally with auto-fetch of the cache.** The dashboard server reads everything (matchups, synergies, personal data, etc.) from `data/analytics_cache.json` — *the SQLite DB is not needed locally*. Sync just that one small JSON file (~few MB) and launch:

```bash
# From your laptop (assumes you have an `~/.ssh/config` Host alias 'brawl')
cd /path/to/local/brawlstar-agent
PYTHONPATH=src uv run python scripts/dashboard.py --remote-cache brawl
```

That single command:
1. rsyncs `data/analytics_cache.json` from the droplet (sub-second on a small file)
2. Launches the local dashboard server on `localhost:8765`
3. Auto-opens your browser

If rsync fails (offline, VPN, droplet down), it falls back to whatever local cache file you have, with a warning. So the dashboard always launches; you just see staler data.

The dashboard header shows freshness:
- `analytics cached 23 min ago · compute took 87s` — healthy
- `analytics cached 4.5h ago` (orange) — stale (timer may have failed; check `systemctl --failed` on the droplet)
- `compute took 32 min` (orange) — compute is slowing down (DB grew, missing index, etc.)
- `compute took 47 min` (red) — likely killed by `TimeoutStartSec=2700`; check `journalctl -u brawl-analytics`
- `computed inline (no cache)` (orange) — no cache was found anywhere; precompute hasn't run yet

To skip the auto-fetch (use whatever local cache exists), just omit `--remote-cache`. To force fresh computation (must have the DB locally too), use `--no-cache` or `--recompute`.

**Alternative 1: SSH tunnel + run dashboard on droplet.** Useful when you're not at your main laptop. Same `localhost:8765` story but on the droplet's hardware:

```bash
ssh -L 8765:localhost:8765 brawl \
  'cd ~/brawlstar-agent && PYTHONPATH=src ~/.local/bin/uv run python scripts/dashboard.py --no-open'
# Browser: http://localhost:8765
```

> The absolute `~/.local/bin/uv` path is necessary because non-interactive SSH shells don't load `~/.bashrc`. Same reason systemd unit files use the absolute path.

**Alternative 2: rsync the DB and run dashboard locally with `--no-cache`.** For ad-hoc deep-dives where you want full power for slicing the raw data with custom queries:

```bash
rsync -avz --progress brawl:brawlstar-agent/data/brawlstars.db \
  /path/to/local/data/brawlstars.db
PYTHONPATH=src uv run python scripts/dashboard.py --no-cache
```

> **Portraits**: the dashboard tries to load brawler portraits from `datasets/character_refs/`. That directory is gitignored. On a fresh laptop, you'll have it. On the droplet, sync once with `rsync -avz --mkpath ...character_refs/ brawl:brawlstar-agent/datasets/character_refs/` so the SSH-tunnel alternative renders portraits too.

For sharing the dashboard outside your laptop (phone, friends), do **not** open UFW. Use Cloudflare Tunnel instead — it exposes the dashboard via a public Cloudflare hostname while keeping your droplet's firewall fully closed. See [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) when you're ready.

### Backup (when set up)

```bash
# On droplet, weekly cron or manual:
sqlite3 ~/brawlstar-agent/data/brawlstars.db ".backup /tmp/brawl.db" \
  && zstd -19 -o /tmp/brawl.db.zst /tmp/brawl.db \
  && rclone copyto /tmp/brawl.db.zst r2:bucket/$(date +%Y-%m-%d).db.zst \
  && rm /tmp/brawl.db /tmp/brawl.db.zst
```

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `ssh: Connection reset by peer` during kex | fail2ban banned home IP, or transient network blip | Web console → `fail2ban-client unban <ip>`. Verify allowlist in `jail.local`. |
| API returns `403 accessDenied` | Outbound IP doesn't match BS API key whitelist | `curl -s ipify.org` on droplet → add to allowed IPs in dev portal |
| `uv sync` fails with `Permission denied` on cache dir | Old `[tool.uv] cache-dir` in `pyproject.toml`, or wrong `UV_CACHE_DIR` | Override with `export UV_CACHE_DIR=$HOME/.cache/uv` |
| `systemctl start` hangs forever on a oneshot service | Normal — oneshot blocks until run completes | Use `--no-block`, or run from another shell |
| `apt install` blocks for ~15 min | needrestart restarted `brawl-collect.service` | Apply Step 7 (`needrestart` config). Stop the running service to unblock apt. |
| `fail2ban-client: Failed to access socket path` after restart | Race with daemon startup | `sleep 2` and retry |
| Crawler loses progress on apt install | needrestart auto-restart | Apply Step 7 |
| DB file getting big | Expected. Plan: partition by month and offload cold months to R2 once >30 GB. Migrate to Postgres only if SQLite stops fitting (DEC-007). | — |

## Migration checklist (changing VPS provider)

If you outgrow DO or want to move providers, the migration is:

1. Provision a new VPS at the new provider (same OS family if possible)
2. Run through Steps 1-15 of this runbook
3. **Before** flipping over: `pg_dump`-style copy of the DB from old → new (`rsync` on a stopped service, or `sqlite3 .backup` on a running one)
4. Update BS API key whitelist with the new IPs (keep old IPs whitelisted during cutover)
5. Once new VPS confirmed healthy: stop the old VPS's timer, remove old IPs from BS API whitelist, destroy old VPS

The local-primary git workflow (DEC-008) means you don't need to migrate any code state — only the gitignored DB and `api.env`. If your DB is huge, consider running a quick `--collect-only` on the new VPS first to populate fresh data, then merging old DB rows in after.

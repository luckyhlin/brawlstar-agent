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

## 14. systemd service + timer for periodic crawls

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
systemctl list-timers brawl-collect.timer --no-pager
systemctl status brawl-collect.service --no-pager
journalctl -u brawl-collect.service -n 50 --no-pager
journalctl -u brawl-collect.service -f             # follow live
sudo fail2ban-client status sshd
```

### Tune cadence

Edit `/etc/systemd/system/brawl-collect.service` to change `--battlelog-limit` or `--rps`, then:

```bash
sudo systemctl daemon-reload
```

(No restart needed; next timer fire picks up the new args.)

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

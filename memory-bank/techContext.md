# Technical Context

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
| numpy | 2.4.4 |
| yt-dlp | latest |
| tesseract | installed (apt) |
| ffmpeg | 4.4.2 |
| Docker | 28.0.1 |

## Production Droplet (DigitalOcean)

| Property | Value |
|----------|-------|
| Provider / plan | DigitalOcean Basic, $6/mo, US region |
| OS | Ubuntu 24.04 LTS |
| Resources | 1 vCPU / 1 GB RAM / 25 GB SSD |
| Reserved IPv4 | `209.38.4.212` (inbound only — SSH lands here) |
| Anchor IPv4 | `64.23.171.86` (outbound source IP — what BS API sees) |
| User | `lin` (sudo), key-only SSH; root login disabled |
| Code path | `/home/lin/brawlstar-agent/` |
| DB path | `/home/lin/brawlstar-agent/data/brawlstars.db` |
| Cron | `brawl-collect.timer` every 6h → `brawl-collect.service` |
| Backup | (planned) Cloudflare R2 nightly via `rclone` |

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

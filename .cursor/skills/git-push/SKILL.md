---
name: git-push
description: >-
  Safely git add, commit, and push. Checks for large files before staging to
  avoid bloating the repo. Use when the user asks to commit, push, or says
  "git push".
---

# Git Push (Safe)

## Workflow

1. **Scan for large files** before staging:
   ```bash
   git add --dry-run . 2>&1 | sed "s/^add '//" | sed "s/'$//" | xargs du -sh | sort -rh | head -20
   ```
   If anything is over 1MB, do NOT stage it — warn the user and suggest adding it to `.gitignore`.

2. **Stage and commit**:
   ```bash
   git add .
   git commit -m "<message>"
   ```

3. **Push**:
   ```bash
   git push
   ```

## Rules

- Never stage files >1MB without explicit user approval.
- Never commit `.env`, credentials, or secrets.
- Check `.gitignore` covers large dirs (`data/`, `datasets/`, `emulator/`, `capture/`, `.venv/`).
- Write a concise commit message that describes the *why*, not the *what*.

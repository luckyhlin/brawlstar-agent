# Kickoff prompt for a fresh agent

> Paste the block below into a new chat. The new agent will read the
> handoff + memory-bank and then ask you which task to start, or proceed
> with the highest-priority pending item.

---

## Prompt to paste

```
You're picking up an in-flight project: a Brawl Stars brawler-pick recommender.
The previous session ran out of context window. The repo is at
/media/lin/disk2/brawlstar-agent/ on this machine.

Before doing anything else, read these files in order:

1. /media/lin/disk2/brawlstar-agent/HANDOFF.md
2. /media/lin/disk2/brawlstar-agent/memory-bank/memory-bank.md
3. /media/lin/disk2/brawlstar-agent/memory-bank/activeContext.md
4. /media/lin/disk2/brawlstar-agent/memory-bank/progress.md  (Session 8 entries
   are most relevant; they describe the recent cold-start + v2 Run A)
5. /media/lin/disk2/brawlstar-agent/memory-bank/decisions.md  (especially DEC-010)
6. /media/lin/disk2/brawlstar-agent/docs/recommender-v1.md

Then proceed with task #1 in HANDOFF.md's "What to do, in priority order" —
which is **Run C** (30-day-window training of the recommender). Save outputs
to models/recommender_v2_30d.* and reports/recommender_v2_30d.json.

Constraints worth keeping front-of-mind:

- DEC-010: do not try to "recover" pre-fix legacy battle labels. They're gone.
  After the cold-start (Session 8), every row in the DB is post-fix-ingested.
- DEC-009: training and inference run on local only. The droplet is 1 vCPU /
  1 GB RAM and is for crawling only.
- DEC-008: code flows local → GitHub → droplet pull. Never edit code on droplet.
- Local DB is 18.6 GB (4M battles). It's expected to be large; don't try to
  shrink it without a plan (schema migration is v3 work).
- The droplet is currently disk-constrained and the user is shrinking it
  manually. If you ssh in, fish auto-launches interactively; see
  docs/deployment.md § 16 for the gotcha table.

When you finish each task, update memory-bank/progress.md (append to
Session 8 continued or open Session 9). Commit incrementally per DEC-008
but don't push without explicit user approval.

Ask me before doing anything destructive (DELETE, VACUUM, DROP, force-push,
running on droplet, etc.).
```

---

## What the new agent will see when they run

After reading those files they'll know:

- The pipeline (`scripts/train-recommender.py`, `scripts/eval-topk.py`) is in
  place and proven on Run A.
- v1 numbers (AUC 0.730 / hit@1 0.150) and Run A numbers (AUC 0.7382 / temporal
  0.7281) are in `progress.md` and `HANDOFF.md`.
- DEC-010 explains why we don't touch legacy data.
- The cold-start + disk-full saga is documented but not blocking — local has
  the full DB, droplet shrinkage is your problem to track separately.

If they need to ssh into the droplet for any reason, the connection details
are in `memory-bank/techContext.md` (`ssh brawl` is the alias). Fish gotchas
are in `docs/deployment.md` § 16.

If you'd rather they pick a different starting task, edit the prompt's
"Then proceed with task #1..." line to point at task #2-6 of the handoff.

# How the three of us work on this repository

Repository language is **English**: file names, code, comments, commit messages, issues.
The report and the oral presentation are in French — that is the only exception.

## Golden rule

`main` must always run. We commit **directly to `main`** — no branches, no pull requests.
That decision only holds if the four rules below are followed every single day.

## The four rules

1. **Pull before you start.** GitHub Desktop → `Pull origin`, at the beginning of every working
   session, even when you are sure nobody pushed anything.
2. **Commit and push as soon as a unit of work is finished.** Not at the end of the day, not "once
   it is clean". Unpushed work is work the others cannot use and you can still lose.
3. **Never push a script that does not run.** `main` is our only shared state — if it is broken,
   the other two are blocked. Run your script once before committing.
4. **Only touch your own files.** If you need a change in a file you do not own, ask in
   `#method-help`. Do not edit it yourself.

**The one exception:** if you really must modify a file you do not own (this will happen once or
twice, near the end, on the report or on assembled figures), create a branch for it that day and
open a pull request. Everything else goes straight to `main`.

## File ownership

One owner per file. The owner is the only person who edits it.

| Area | Files | Owner |
|---|---|---|
| Data, description, unsupervised | `01_scripts/data_*.py`, `02_data/`, `01_scripts/clustering.py` | *Emma* |
| Supervised modelling | `01_scripts/models.py`, `05_results/` | *Teodora* |
| Orange3 and report | `07_orange/`, `06_report/`, `08_slides/` | *Nahi* |

## Commit messages

English, imperative mood, one idea per commit. Summary line under ~60 characters, no final period.
Add `closes #N` when the commit finishes an issue — the issue closes and its board card moves on
its own.

```
add Lasso feature selection to the model comparison

closes #13
```

Good: `add country-level working table script`, `fix missing-value imputation inside pipeline`.
Bad: `update`, `fix`, `wip`, `changes`.

## Code review without pull requests

Once a week, each of us opens someone else's commit on GitHub (**Commits** tab → click a commit)
and leaves a comment on at least one line. The diff view allows line-level comments. Five minutes,
and it replaces what a pull request review would have given us.

## What we never commit

Raw data, large files, regenerable outputs (figures, result CSVs), virtual environments, caches.
All of it is already listed in `.gitignore`. The download URL and extraction date of the raw data
are documented in `02_data/README.md` so that anyone can rebuild the file.

## Files that merge badly

- **`.ows` (Orange3)**: generated XML, unreadable in a conflict. One file per person
  (`07_orange/workflow_<firstname>.ows`), and nobody edits somebody else's file. Workflows are
  merged at the end of the project, in pairs, on a single machine.
- **Jupyter notebooks**: outputs pollute diffs. Notebooks in `03_notebooks/` are for throwaway
  exploration only; code that matters ends up in `01_scripts/` as functions. Before committing a
  notebook: `jupyter nbconvert --clear-output --inplace my_notebook.ipynb`.
- **The report**: written collaboratively in Google Docs / Overleaf. Only the exported PDF is
  committed, in `06_report/`. A `.docx` in a Git conflict is unrecoverable.

## Merge conflicts

They are rare with one owner per file, and they happen when two people edit the same file. If Git
reports one, do not force anything: open the file, both versions are marked, keep both logics, save,
commit. Never run `git push --force`.

## Project tracking

- One GitHub issue per task, assigned to one person, attached to a milestone
  (`Jalon 1` / `Jalon 2` / `Jalon 3`).
- Board view in **Projects**: move your card to *In Progress* when you **start**, not when you
  finish.
- One tag per milestone delivered: `git tag -a jalon1 -m "milestone 1 delivered" && git push --tags`.
- Discord carries the flow, GitHub carries the state: any decision taken in a channel is written
  into the repository the same day.

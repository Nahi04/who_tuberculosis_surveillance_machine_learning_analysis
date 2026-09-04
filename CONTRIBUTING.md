# How the three of us work on this repository

## Golden rule

`main` must always run. Never push straight to `main`: go through a branch + a pull request (PR)
reviewed by another member.

## Daily cycle

```bash
git checkout main
git pull                          # always before starting
git checkout -b feat/svm-comparison   # one branch = one task
# ... write code ...
git add 01_scripts/models.py
git commit -m "add RBF SVM to the model comparison"
git push -u origin feat/svm-comparison
```

Then on GitHub: *Compare & pull request* -> assign a reviewer -> once approved, *Squash and merge*
-> delete the branch.

Branch naming: `feat/...` (new work), `fix/...` (bug fix), `doc/...` (writing), `orange/...`
(Orange3 workflow).

Commit messages: imperative, short, one idea per commit. `add Lasso feature selection`, not
`update`.

## What we never commit

Raw data, large files, regenerable outputs (figures, result tables), virtual environments, caches.
All of it is already listed in `.gitignore`. If a data file exceeds ~50 MB and really must be
versioned, use Git LFS; otherwise document its download URL in `02_data/README.md`.

## Files that merge badly

- **`.ows` (Orange3)**: generated XML, unreadable in a conflict. One file per person
  (`07_orange/workflow_<name>.ows`), and never edit someone else's file in parallel. Workflows are
  merged at the end of the project, in pairs, on a single machine.
- **Jupyter notebooks**: stored outputs pollute diffs. Keep notebooks in `03_notebooks/` for
  throwaway exploration only; code that matters ends up in `01_scripts/` as functions. Before
  committing a notebook: `jupyter nbconvert --clear-output --inplace my_notebook.ipynb`.
- **The report**: write it collaboratively in Google Docs / Overleaf and commit only the exported
  PDF into `06_report/`. A conflicted `.docx` cannot be recovered.

## Resolving a merge conflict

```bash
git checkout my-branch
git pull origin main      # bring main into your branch
# git lists the conflicting files; edit them, keep both logics
git add <resolved file>
git commit
git push
```

When in doubt: never run `git push --force` on a shared branch.

## Project tracking

- One GitHub issue per task, attached to a milestone (`Milestone 1/2/3`) and assigned to a person.
- One tag per delivered milestone: `git tag -a milestone1 -m "milestone 1 submitted" && git push --tags`.
- The *Projects* board view is more than enough for three people over three weeks.

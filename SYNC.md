# Keeping local and GitHub in sync

## Before you start working

```powershell
cd c:\GitHub\Hallucinations
git pull origin main
```

## After you make changes

```powershell
git status
git add -A
git commit -m "Describe your change"
git push origin main
```

## If GitHub has changes you don’t have yet

If `git push` is rejected:

```powershell
git pull origin main --rebase
git push origin main
```

## If you only want to match the remote (discard local uncommitted edits)

```powershell
git fetch origin
git reset --hard origin/main
```

⚠️ This deletes uncommitted local changes.

## Branches

Other branches (e.g. `v1`, `v2`) can be updated the same way with `git pull origin <branch>` / `git push origin <branch>`.

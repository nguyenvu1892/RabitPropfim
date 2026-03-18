---
description: Git branching workflow for RabitPropfim DRL project
---

# Git Branching Workflow

// turbo-all

## Rules
- **NEVER code directly on `main`**. Always create a feature branch.
- Branch naming convention: `sprint{N}/T{task_id}-{short-description}`
  - Example: `sprint1/T1.1.1-init-monorepo`
  - Example: `sprint2/T2.1.1-physics-sim`

## Starting a new task

1. Make sure you're on the latest main:
```bash
git checkout main
git pull origin main
```

2. Create a new branch for the task:
```bash
git checkout -b sprint{N}/T{task_id}-{short-description}
```

3. Do your work, commit frequently with clear messages:
```bash
git add .
git commit -m "feat(module): description of change"
```

4. Push branch to remote:
```bash
git push origin sprint{N}/T{task_id}-{short-description}
```

## Commit message format
- `feat(module): add new feature`
- `fix(module): fix bug`
- `test(module): add tests`
- `docs: update documentation`
- `chore: project setup, configs`

## After task is done
- Create PR to main (or merge locally if solo dev)
- Ensure all tests pass before merging

## DEVLOG Rule (BẮT BUỘC)
- **Mỗi thay đổi** trên dự án PHẢI ghi vào `DEVLOG.md` ở root project
- Format: `### [LOẠI] Mô tả ngắn` + danh sách bullet những gì đã làm/sửa
- Loại: `[PLAN]`, `[FEAT]`, `[FIX]`, `[TEST]`, `[REFACTOR]`, `[CONFIG]`, `[DOCS]`
- Ghi ở đầu file (mới nhất ở trên), nhóm theo ngày

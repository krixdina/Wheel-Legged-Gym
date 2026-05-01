# Git Commit After Code Changes

If this round of work produces any functional code changes, create a git commit before ending the task.

## Rules
- Only commit when there are actual functional code changes. 
- **Exception for Comment-Only Changes:** If the modifications consist *entirely* of adding, modifying, or deleting code comments without altering any executing code or logic, **DO NOT create a git commit**. Instead, only output the summary of your comment changes directly in your chat response using this exact format: `<本轮注释修改摘要>`.
- Do not create empty commits.
- Commit all files that belong to the current round of work.
- Use a concise English commit message that accurately describes the main change.
- If multiple related changes happen in the same round, combine them into one commit.
- If changes are incomplete, blocked, or not yet coherent, explain the reason instead of forcing a bad commit.

## Commit message format (For functional code changes only)
<类型>: <本轮修改摘要>

## Type examples
- feat
- fix
- refactor
- docs
- test
- chore
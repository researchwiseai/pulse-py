# Development Workflow

## Task Completion Protocol

**CRITICAL**: At the end of every task, always commit and push changes to maintain project history and enable collaboration.

### Required Steps for Task Completion
1. Complete the requested work
2. Verify the implementation works as expected
3. Stage all changes: `git add .`
4. Commit with descriptive message: `git commit -m "descriptive message"`
5. Push to remote: `git push`

### Commit Message Guidelines

- Use Present Tense
  Write in present tense (e.g., “Add feature” not “Added feature”).

- Use Multi-Line Messages
  Favor multi-line commit messages with a clear structure:

   1. Short Summary (first line, ≤ 50 characters) – describes the core change.

   2. Detailed Description (body) – explain why the change was made, not just what it does.

   3. Plan Changes – list any deviations from the original plan or design decisions.

   4. New Tech / Dependencies – mention any libraries, patterns, or tools introduced.

   5. TODOs & Placeholders – explicitly note any follow-up work left behind (even if just stubs or placeholders).

- Reference Issues
  Include links or references to relevant issues, tickets, or discussions when possible.

- Be Descriptive
  Aim for clarity and completeness — future you (and reviewers) should understand why this commit exists without reading the diff.


### When to Commit
- After completing any requested task
- After fixing bugs or issues
- After adding new features or functionality
- After updating documentation or configuration
- Before switching to a different task or feature

### Exception Cases
- If explicitly told not to commit by the user
- If working on experimental changes that shouldn't be persisted
- If the changes break existing functionality (fix first, then commit)

This workflow ensures that all progress is tracked, changes are backed up, and team members can see the latest updates.


USE CONTEXT7 ON EVERY TASK
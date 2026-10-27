# Lessons

## Match the project's commenting pattern (2026-10-24)

Correction: comments and docstrings I added were long explanatory paragraphs, and new files lacked the standard header.

Rules:
- Every new `.py` source file starts with the header docstring: file name, `Author: Aritra Roy`, `Email: contact@aritraroy.live`, `Website: https://aritraroy.live`, `GitHub: aritraroy24` (the user normalised all source headers to this in d9a1614, replacing `Date:`). Test files still use the `Date: DD-MM-YYYY` line.
- Inline comments are short and terse, like `# Check if data is directly accessible`: a few words, no trailing period, no rationale paragraphs.
- Docstrings: a one-line summary, then an `Args:` block with `name (type, required|optional): description` entries for every argument, and a `Returns:` block with `type: description` (the style the user retrofitted across the codebase in d9a1614). No multi-paragraph design explanations; those belong in commit messages or `tasks/todo.md`.
- Before committing, grep the diff for added `#` comments and docstrings and compare them against neighbouring code.

## Docs are reviewed before committing

Correction: I committed a large docs update without asking.

Rules:
- Leave docs changes (docs/, README, changelogs, blog posts) uncommitted and ask before committing; the user reviews them manually first.
- Never add a Claude co-author line to commit messages.

# Lessons

## Match the project's commenting pattern (2026-10-24)

Correction: comments and docstrings I added were long explanatory paragraphs, and new files lacked the standard header.

Rules:
- Every new `.py` file (source and tests) starts with the header docstring: file name, `Author: Aritra Roy`, `Email: contact@aritraroy.live`, `Website: https://aritraroy.live`, `Date: DD-MM-YYYY`.
- Inline comments are short and terse, like `# Check if data is directly accessible`: a few words, no trailing period, no rationale paragraphs.
- Docstrings use the existing style: a one-line summary, then an `Args:` block with `name (type, optional): description` entries. No multi-paragraph design explanations; those belong in commit messages or `tasks/todo.md`.
- Before committing, grep the diff for added `#` comments and docstrings and compare them against neighbouring code.

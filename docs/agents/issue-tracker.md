# Issue tracker: GitHub

Issues and specs live in `SirDarcanos/minires-models`.
Use the `gh` CLI from this clone; it infers the repository from the remote.

- Publish a ticket: `gh issue create --title "..." --body "..."`
- Read a ticket: `gh issue view <number> --comments`
- Inspect its fields: `gh issue view <number> --json title,body,labels,comments`
- List tickets: `gh issue list --state open`
- Comment: `gh issue comment <number> --body "..."`
- Update labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

Inspect ticket labels alongside its body and comments. Use heredocs with
`--body-file -` for multiline bodies. When a skill says to publish to the
issue tracker, create a GitHub issue; when it says to fetch a ticket, read
its body, labels, and comments.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares issue and pull-request numbers. Resolve an ambiguous number
with `gh pr view <number>` and fall back to `gh issue view <number>`.

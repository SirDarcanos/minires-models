# Repository labels

## Triage status

Canonical skill roles map directly to these tracker labels.

| Role | Label | Meaning |
| --- | --- | --- |
| needs-triage | `needs-triage` | Maintainer evaluation needed |
| needs-info | `needs-info` | Waiting on reporter information |
| ready-for-agent | `ready-for-agent` | Fully specified for autonomous implementation |
| ready-for-human | `ready-for-human` | Requires human implementation |
| wontfix | `wontfix` | Will not be actioned |

Use at most one triage-status label per issue. When changing status, remove
the previous status label. `wontfix` is a terminal resolution; close the issue
with an explanation when applying it.

## Issue type and resolution

| Label | Meaning |
| --- | --- |
| `bug` | Something isn't working |
| `documentation` | Improvements or additions to documentation |
| `enhancement` | New feature or request |
| `question` | A question about usage or behavior |
| `duplicate` | This issue or pull request already exists |
| `invalid` | This doesn't seem right |

Type labels may coexist with a triage status. `question` describes the issue;
`needs-info` means progress is waiting on the reporter. When resolving an
issue as `duplicate`, link the original. Explain an `invalid` resolution
before closing; use `needs-info` while clarification could establish validity.

## Contributor help

| Label | Meaning |
| --- | --- |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |

These labels may coexist with issue-type and triage-status labels.

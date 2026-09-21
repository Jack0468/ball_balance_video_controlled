---
name: weekly-report
description: Generates Jack McCudden's weekly progress report (as Markdown, ready to paste into Word) for the VRI 2026 project, matching his established employment/research reporting format exactly. Trigger this whenever the user asks for a "weekly report," "progress report," "status report," or says things like "write this week's report," "do the weekly," or mentions report_weekly, even if they don't name the skill directly — this is a real recurring obligation, not an optional writeup. Also trigger if the user shares a OneNote export/PDF of weekly notes and asks it to be turned into a report.
---

# VRI 2026 weekly report

Jack has filed a weekly progress report in this exact three-part format
every week since June 2026. The goal here is reproduction, not improvement —
read `references/format-spec.md` before drafting anything; it holds two full
real examples pulled directly from his actual `.docx` files. Match the tone
(terse log-line bullets in Summary, technical prose paragraphs in Weekly
Record, concrete bullets in Work Next) and don't drift toward corporate
status-report phrasing.

## Why this can't be built from git alone

One of the two real examples in the format spec (2026-09-11) is almost
entirely hardware/admin work — Jetson flashing, SSH setup, HR training —
none of which appears in git history. Git and the project logbook cover the
*code* side of the week; they systematically miss everything else. Always
ask the user for that week's notes (a OneNote PDF export, or just typed
notes in chat) before drafting — don't assume a quiet git log means a quiet
week.

## Process

**1. Determine the reporting week.**
List the filed-reports folder to find the most recent one, and infer the
next Monday–Friday range from it:
```
ls "C:\Users\Admin\OneDrive - The University of Sydney (Students)\.research\EMPLOYMENT_INFO\report_weekly\word\"
```
(filed `.docx` reports moved here as of 2026-09-18 — if it's empty or
missing, also check the parent `report_weekly\` folder for older files not
yet migrated.)
(Listing works fine directly on this OneDrive path even though *reading
file contents* there has hit permission errors before — not an issue here
since the deliverable is Markdown, not a file read from that folder.)
Confirm the inferred date range with the user rather than assuming — they
may be catching up on a skipped week or reporting something mid-week.

**2. Pull git history for the range.**
```
git log --since="<range start>" --until="<range end + 1 day>" --all --date=short --pretty=format:"%h %ad %s"
git log --since="<range start>" --until="<range end + 1 day>" --stat
```
Run from the repo root. Use this to identify which modules/workstreams were
actually touched (cross-reference directory names against CLAUDE.md's
"Current State" table and `.claude/agents/*.md` so the report uses the
project's own vocabulary — e.g. "Track 1", "Arm 2" — instead of inventing
new names for the same thing).

**3. Pull the logbook entries for the range.**
`docs/PROJECT_LOGBOOK.md` is newest-first. Grep for `^## DD/MM/YYYY` headings
falling inside the range and read those sections — this is where the *why*
behind a change lives, which raw git log won't give you.

**4. Pull the week's OneNote export.**
Weekly OneNote PDF exports live in
`C:\Users\Admin\OneDrive - The University of Sydney (Students)\.research\EMPLOYMENT_INFO\report_weekly\pdf\`,
named `<year>_<ISO week number>.pdf` (e.g. `2026_38.pdf`). Compute the ISO
week number for the reporting range and read that file with the `pdf`
skill — this is where admin/hardware/meeting activity lives that git and
the logbook systematically miss (see above). To get the ISO week number:
```
"C:/Users/Admin/.conda/envs/ball_balance_env/python.exe" -c "import datetime; print(datetime.date(<year>,<month>,<day>).isocalendar().week)"
```
using the Monday of the reporting range. If no matching PDF exists for that
week, ask the user directly rather than silently skipping this step —
non-code activity has been the majority of some past reports (e.g.
2026-09-11) and omitting it produces a materially incomplete report.

**5. Check the previous filed report's Work Next for carryover items.**
Read the most recent filed report (step 1) and compare its Work Next
bullets against this week's sources. An item doesn't disappear just because
nothing new happened on it this week — e.g. "Design enclosure/tripod" sat
untouched in git/logbook/onenote for two straight weeks (09/11 and 09/18)
but stayed in Work Next both times because it's genuinely still open
(CLAUDE.md's Hardware row is "Not started"). Carry an item forward unless
the user confirms it's done or superseded; dropping it just because this
week's diff doesn't touch it is a real gap, caught the first time this
skill was actually run (2026-09-18 trial against the real 15-18/09 report).

**6. Draft the three sections and deliver as Markdown.**
Jack pastes the output into Word himself and finishes formatting there, so
the deliverable is plain Markdown, not a generated `.docx` — no document
tooling, no OneDrive write, no local build step needed. Output it directly
in the chat response using this shape (mirrors `references/format-spec.md`
exactly):

```markdown
**Weekly Progress Summary**

Date: DD/MM/YYYY~DD/MM/YYYY
Name: Jack McCudden

**Summary**
- bullet
- bullet

**Weekly Record**

**<Workstream heading>:**

<prose paragraph>

**<Workstream heading>:**

<prose paragraph>

**Work Next**
- bullet
- bullet
```

The blank line between each heading and its paragraph is required, not
stylistic — Markdown only breaks a new paragraph on a blank line, and the
real filed reports have the heading and prose as genuinely separate
paragraphs (confirmed via `pandoc -t markdown` on the source `.docx`, see
`references/format-spec.md`). A single newline with no blank line renders
the heading and prose as one run-on line once pasted into Word — a real bug
caught in this skill's first trial run.

Keep Weekly Record paragraphs specific: numbers, results, concrete
technical actions, not "continued working on X." This is a real report
someone else reads, not a first-draft dump — if anything in git/logbook is
ambiguous (e.g. a commit that touches multiple workstreams, or a vague
commit message), ask rather than guessing at what to say happened.

Save the final Markdown to
`C:\Users\Admin\OneDrive - The University of Sydney (Students)\.research\EMPLOYMENT_INFO\report_weekly\md\weekly_progress_<YYYY_MM_DD>_jack.md`,
using the same Friday-end-date naming convention as the filed `.docx`
reports. This mirrors the `word\`/`pdf\` split Jack set up in that folder
on 2026-09-18. Still paste the content directly in the chat response too —
that's what actually gets copied into Word — the file is a durable record,
not a replacement for showing it in chat.

# Weekly report format spec

This is Jack McCudden's real weekly progress report format for the VRI 2026
research employment, in continuous use since June 2026 (~12 weeks of examples
in the OneDrive `report_weekly` folders). It is not a template to redesign —
match it. The structure below was extracted directly from real `.docx` files
via `pandoc -t markdown`, not paraphrased from memory.

## Structure

1. **Title** (bold): `Weekly Progress Summary`
2. `Date: DD/MM/YYYY~DD/MM/YYYY` — the Monday~Friday of the reporting week
3. `Name: Jack McCudden`
4. **Summary** (bold heading) — 2-4 short bulleted one-liners, one per major
   workstream touched that week. Terse, mixed present/past tense. Not full
   sentences with subjects always spelled out — reads like a log line.
5. **Weekly Record** (bold heading) — one bold sub-heading per workstream,
   each followed by a *prose paragraph* (not bullets) narrating what was
   actually done. Technical, specific, includes real numbers/results when
   there are any. Reads like an engineering log entry, not a status-report
   platitude ("made progress on X" is too vague — "lowered the saturation
   floor to 20, recovering green detection from 0% to 94.5%" is the bar).
6. **Work Next** (bold heading) — 3-4 plain bullets, concrete next actions.

Bullets in the real files are informal (typed characters, not a native Word
list) — deliver as standard Markdown `-` bullets. Jack pastes the output
into Word and finishes formatting there, so plain Markdown is the target
output, not a generated `.docx`.

## Real example (2026-09-18, lightly reformatted from pandoc output)

```
Weekly Progress Summary

Date: 15/09/2026~18/09/2026
Name: Jack McCudden

Summary
- Recorded data of the small model pipeline on the jetson directly.
- Continued medium audio model training
- Designed code infrastructure for large model testing on the Jetson

Weekly Record

Small Model Data Collection on the Jetson:
Ported the core vision system to the Jetson AGX Orin and confirmed reliable
bi-directional serial communication with the STM32. Evaluated the vision
pipeline live and identified some bugs with the colour classification of the
vision pipeline. Lowered the saturation floor to 20 to recover green
detection (improving from 0% to 94.5%). Compiled a report outlining the
Vision Action sections of the system with the jetson deployment.

Medium Audio Model training:
Continued development on the NeMo audio model architecture, specifically
focusing on model accuracy and robustness when subjected to background
noise. Model training was inhibited by google colab usage restrictions.

Large VLA baseline and finetune:
Picked baseline VLA models (Qwen2.5-VL-3B-Instruct, InternVL2.5-4B,
PaliGemma2-3B-mix-448, Moondream2) and ran evaluation on local dataset
before attempting to trial run on hardware. Results will be ready after the
weekend. Ran a successful deployment smoke test for Qwen2.5-VL-3B on the
Jetson (loaded in FP16, generating correct text outputs with a wall-clock
latency of ~183s). Collected additional training data for the VLA model.
Generated a process to fine tune Qwen on our data set and action space. No
computation has been achieved on this due to colab constraints.

Work Next
- Run evaluation on the selected large model baselines without any
  additional fine tuning.
- Determine alternatives to colab for ml training and evaluation testing.
- Design enclosure / tripod for demonstration system which encapsulates the
  Jetson
```

## Real example (2026-09-11 — a week with almost no code activity)

This example matters because it proves the report is NOT purely a git-log
summary: an entire week can be dominated by hardware/admin work that never
touches the repo.

```
Weekly Progress Summary

Date: 7/09/2026~11/09/2026
Name: Jack McCudden

Summary
- Successfully performed a hard direct flash of the Nvidia Jetson hardware
  using a dedicated Ubuntu installation.
- Established a consistent ethernet SSH connection between the Jetson and a
  Windows laptop, and installed required dependencies.
- Completed necessary Workday training modules.

Weekly Record

Nvidia Jetson Flashing & Configuration:
Installed Ubuntu directly onto a spare personal laptop to execute a hard
direct flash of the Jetson, bypassing the Nvidia SDK manager entirely. Used
a monitor for the initial account setup and manually found dependencies
using a mobile hotspot connection. Troubleshot and established a consistent
ethernet SSH connection to the Windows laptop that successfully persists
through reboots. Final dependency requirements were installed onto the
Jetson via SSH control from the Windows host.

Administration & Access:
Checked Workday and completed the listed training modules from HR.

Work Next
- Deploy required inference modules to the jetson to collect data for the
  small ML model baselines
- Configure and deploy our researched larger VLA models to the jetson to
  collect data to compare with our benchmarks
- Design enclosure / tripod for demonstration system which encapsulates the
  Jetson
```

## Naming convention (for reference / if ever saving a file)

The final filed `.docx` (which Jack produces himself in Word from the
pasted Markdown) follows `weekly_progress_YYYY_MM_DD_jack.docx`, where
`YYYY_MM_DD` is the **Friday (end date)** of the reporting week, zero-padded
(e.g. `09_18`, not `9_18` — some historical files in the older
`VRI_2026_WINTER` folder used unpadded months; the current
`EMPLOYMENT_INFO` series is zero-padded going forward). Filed `.docx` copies
live in
`C:\Users\Admin\OneDrive - The University of Sydney (Students)\.research\EMPLOYMENT_INFO\report_weekly\word\`
(as of 2026-09-18 — previously the bare `report_weekly\` folder; check both
when looking for the most recent filed report until older files are
migrated). The week's OneNote PDF export lives in the sibling `pdf\`
subfolder, per the naming convention in SKILL.md step 4.

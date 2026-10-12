# FPGA ↔ Photonic Platform Interface Spec — FIRST DRAFT (Arm 3 Bridge)

> **Status: DRAFT, NOT A SPEC YET.** This document exists to make the Arm 3 blocker
> concrete and actionable — it converts "no interface spec exists" into a specific list of
> what's known, what's assumed, and what's genuinely unknown. It is explicitly **not**
> ready to synthesize RTL against. **Tag key** (used throughout, including §7): `CITED` =
> directly attributable to the source paper via the notes file; `DERIVED` = arithmetic done
> by this agent on top of a cited number, not itself a paper spec; `PROPOSED` = a reasonable
> default pattern-matched from this project's existing conventions, not yet decided by
> anyone with authority to decide; `UNKNOWN` = a genuine gap, not guessed at. **Nothing in
> this document is tagged `CONFIRMED` as a project decision, because no project decision has
> been made yet** — see §7 for the full ledger and §8 for what a real spec still needs
> answered, most importantly the framing question in §1, which this draft cannot resolve on
> its own.

**Author context**: written 2026-10-13, FPGA agent, as background/research work (no Jetson
or physical-hardware access used or needed). Source material: `docs/research/photonic_pnn_component_notes.md`
(gitignored personal research notes, summarizing Sved et al., *Nature Communications* **17**,
1059 (2026) — a real, published, fabricated nanophotonic neural-network accelerator; read in
full before writing this draft). Current FPGA-side context drawn from `fpga/docs/camera_pinout_zedboard.md`,
`fpga/hls_hardware/`, `docs/HARDWARE_AND_SOFTWARE_PREREQUISITES.md`, and `docs/PROJECT_LOGBOOK.md`
(searched for any existing Arm 3 / photonic content — there is none yet; this is genuinely
first-draft territory, not a revision of prior work).

---

## 0. One-paragraph summary

The source paper describes a **passive** photonic accelerator: weights are fixed at
fabrication time, and the only runtime-dynamic electronics are (a) a per-channel analog
voltage write to set each input channel's amplitude and phase before light enters the fixed
optical core, and (b) an optical-power read at each output channel afterward. That maps onto
"DAC array out, ADC/photodetector array in" — a shape an FPGA is well suited to drive. But
the paper's own demonstrated control uses slow thermo-optic actuators, explicitly named by
its own Discussion as too slow for real interfacing, with a named-but-unspecified upgrade
path (EO/plasma-dispersion/Pockels modulators). Nothing in the paper used digital logic at
all — there is no FPGA reference design to adapt, no photodetector/ADC part specified, and no
settling-time number given for either the demonstrated thermo-optic path or the named upgrade
path. **Above all**, this project has not yet answered whether Arm 3 targets a real fabricated
chip, a simulation/co-design-only stand-in, or a commercial photonic accelerator product —
that framing question changes what "interface spec" even means here, and this draft does not
guess at it.

---

## 1. The framing question that blocks a real spec (UNKNOWN — do not guess)

A real interface spec needs to know which of these Arm 3 actually is. Nothing in evidence
available to this agent answers it:

- **(a) An actual fabricated photonic chip** this project has, or will have, physical access
  to (its own device, a university/lab collaborator's device, or a custom fab run) — in which
  case the interface is a real electrical/optical bridge to real silicon, and the DAC/ADC
  channel counts, resolutions, and the laser source all need real part numbers.
- **(b) A simulation/co-design-only arm** — the "photonic platform" is modeled (e.g., in
  Lumerical/FDTD or a numerical surrogate) and the FPGA's role is to interface with a
  simulated optical core over some digital channel, not drive real analog hardware at all. If
  this is the case, most of §3/§4 below (DAC resolution, photodetector specs, thermo-optic
  timing) are the wrong questions entirely — the real spec would be a data format for a
  sim-in-the-loop link, not an analog bridge.
- **(c) A commercial photonic accelerator product** (e.g., a vendor's board-level photonic
  compute accelerator) — in which case the "interface spec" is really "read that vendor's
  datasheet," and nothing in the Sved et al. paper (which is a research fabrication, not a
  product) would apply directly.

**This draft assumes nothing here.** Everything below is written as if (a) were true,
because that is the scenario the source paper actually describes and the only one with
concrete numbers to reason from — but that is a drafting convenience, not a decision, and is
flagged again in §7/§8.

---

## 2. What the source material does and doesn't cover

`docs/research/photonic_pnn_component_notes.md` is a components/interface summary of one
real paper, not a project-authored spec and not an FPGA design reference. Per that file's own
§8 ("What this does NOT tell us"):

- No digital logic (FPGA/ASIC) was used anywhere in the paper's setup — control was a PCB +
  external bench instruments (tunable laser, multi-channel optical power meter) + presumably
  a host PC/MCU sweeping DAC voltages. **There is no existing reference design to adapt.**
- No photodetector or ADC component is specified anywhere in the paper.
- No real-time/closed-loop operation was demonstrated; the device is calibrated once (DAC
  sweep → response table) then run open-loop at inference time.
- No deployed-inference latency/throughput numbers are reported (only training-time FDTD
  simulation wall-clock, which is a design-time cost, not a runtime number).

Any number below that isn't directly attributable to the paper is marked as such.

---

## 3. What the FPGA would need to drive (digital→optical, per-inference)

Per the notes file §4, the demonstrated architecture's input-encoding chain is: split one
input beam N ways, run each of the N channels through an amplitude-control MZI stage and
(for N−1 of them, relative to a reference channel) a phase-control stage, each stage actuated
by a thermo-optic microheater driven by an analog DAC voltage.

### 3.1 Channel count

- **`CITED`**: 2N−1 DAC channels per inference (N amplitude channels + N−1 phase channels,
  channel 1 serves as the phase reference) — directly stated in the notes file §4.
- **`CITED`, example sizes only**: the two *fabricated* devices in the paper are N=10 (MNIST,
  10 in / 10 out → 19 DAC channels, 10 readback channels) and N=15 (MedNIST, 15 in / 6 out →
  29 DAC channels, 6 readback channels). **`UNKNOWN`**: whether any future Arm 3 device (if
  (a) in §1 holds) would be this size, larger, smaller, or tiled (per the notes file §7's
  proposed-but-unfabricated multi-core scaling) — there is no project-specific N/C yet.

### 3.2 Resolution

- **`CITED` + `ASSUMPTION` (reasoning is ours, number is derived from the paper's own
  calibration procedure)**: the notes file reports the calibration sweep covered 1.1 V–5 V in
  3 mV steps (~1,300 distinct levels), implying roughly 11 bits of DAC resolution to match the
  calibration granularity *actually used* in the paper. This is not a stated DAC spec in the
  paper — it's back-calculated from the calibration procedure, and is a lower bound on what
  worked for *that* device's heater response curve, not a general requirement.
- **`CITED`**: the paper separately reports the system tolerates mean phase error up to
  ~0.89–1.18 rad before visible accuracy degradation (notes file §5) — i.e., there is real
  margin, so 11 bits may be conservative, not a hard floor. **`UNKNOWN`**: the actual
  minimum resolution that would preserve accuracy on whatever task Arm 3 ends up running
  (VLA action regression, not MNIST/MedNIST classification) — no data to extrapolate from.

### 3.3 Timing budget

- **`CITED`**: the paper's own Discussion states its thermo-optic control is slow
  ("thermal time constants, not suited to real-time/high-bandwidth operation") and explicitly
  names the upgrade path: "incorporating high-speed modulation schemes, such as those based
  on the plasma dispersion effect, electro-optic effect or Pockels effect, along with fast
  photodetectors and ADCs, can facilitate high-bandwidth data interfacing for the PNNs."
- **`UNKNOWN` — genuine gap, flagged explicitly**: no quantified settling-time number is
  given for the thermo-optic path in the notes file, and none exists yet for the named
  EO/Pockels upgrade path either (it's named as a direction, not specced). This agent is
  **not** going to assert a microsecond/millisecond figure from general photonics-literature
  recall here — that would be exactly the "confidently guessing a spec that doesn't exist"
  failure mode this draft is supposed to avoid. If thermo-optic settling is anywhere in the
  multi-microsecond-to-millisecond range typical of microheater phase shifters generally, it
  could consume a large fraction of this project's 120Hz/8.3ms control-loop budget per
  inference, before DAC settling, optical propagation, or digital readback are even added —
  **this is a risk to flag, not a computed number.**
- **`UNKNOWN`**: whether the 120Hz/8.3ms loop requirement (`CLAUDE.md`, binding on any
  eventual FPGA-hosted production controller, explicitly *not* binding on the Jetson
  comparison arms) binds Arm 3 at all. Arm 3 is not the Jetson arm, so the carve-out doesn't
  obviously apply, but nothing in `CLAUDE.md`, `docs/EVALUATION_STRATEGY.md`, or the logbook
  states a per-arm latency target for the photonic arm specifically. Flagged in §8.

---

## 4. What the FPGA would need to read back (optical→digital, per-inference)

- **`CITED`**: C photodetector/ADC channels, one per optical output port (C=10 for the MNIST
  device, C=6 for MedNIST — example sizes only, see §3.1).
- **`UNKNOWN` — genuine gap, explicitly named in the notes file §8**: "No photodetector or
  ADC component is specified or characterized anywhere in the paper (power readout used a
  laboratory optical power meter) — if Arm 3 needs real photodetector/ADC specs, this is a
  genuine gap this source doesn't fill." Any photodetector/ADC part, bandwidth, or resolution
  number in a future real spec is a **placeholder assumption, not a decision**, until a real
  device/datasheet is chosen.
- **`CITED`**: the paper's own monitoring/feedback taps (10:90 MMI, interferometric
  null-seeking) are explicitly for one-time calibration, not live per-inference feedback —
  the device is calibrated once then run open-loop at inference time.
- **`UNKNOWN` — do not read the previous bullet as "no recalibration path needed"**: the
  notes file §8 separately flags that "drift/recalibration behavior over time is not
  characterized" in the paper. Whether a real deployed bridge needs to periodically
  re-exercise the calibration-monitoring taps (not just the C-channel inference readback) to
  compensate for thermal/phase drift over time is an open question this source doesn't answer
  — not a decision this draft is making either way.

---

## 5. How this would map onto this project's actual FPGA (ZedBoard/Zynq-7000 XC7Z020)

Grounded in what's actually documented for this board, not generic photonics-interfacing
theory. **Everything in this section is conditional on framing (a) in §1** (a real
fabricated/physical chip this project interfaces with electrically) — if (b) or (c) turns out
to be true instead, none of the ZedBoard pin/peripheral planning below applies, and that
caveat is restated rather than assumed away:

- **No on-chip DAC exists on the Zynq-7000.** Any analog output (the 2N−1 DAC channels in
  §3.1) would need an **external** multi-channel DAC — this project has made no part choice
  yet. `fpga/docs/camera_pinout_zedboard.md` confirms **Pmod JC and JD are fully consumed by
  the OV7670 camera wiring** (all 8 data pins per header plus PCLK/XCLK/VSYNC/HREF/SIOC/SIOD).
  Remaining headroom for a DAC/ADC daughter interface would be **Pmod JA, JB, JE**, the
  dedicated **XADC header**, or the high-pin-count **FMC-LPC connector** — but this agent does
  not have a verified, in-session-confirmed pinout/channel-count for the ZedBoard's XADC
  header breakout, so no channel count is asserted here (`UNKNOWN`, to confirm against the
  ZedBoard reference manual before any real pin planning).
- **The Zynq does have an on-chip XADC** (a 12-bit SAR ADC block built into the PS, with
  multiple auxiliary channels available via external header pins) — this is a real, existing
  on-chip resource that could plausibly serve as part of the C-channel readback path (§4) if C
  is small enough to fit the auxiliary channels actually broken out on this board, **but** the
  exact number of ZedBoard-exposed XADC channels is the same unverified fact flagged above —
  not asserted from memory here.
- **Resource budget**: unlike the paused vision-weight-compilation work, this bridge's
  on-chip cost is mostly a control FSM plus per-channel calibration-table storage (2N−1 + C
  entries, tiny compared to 612.5 KB BRAM) and whatever SPI/parallel bus logic talks to the
  external DAC/ADC — **DSP/BRAM budget is very unlikely to be the constraint here; the timing
  budget in §3.3 is the real open risk.**
- **Interface convention precedent**: Domain 4's in-progress "Option B" motor-control port
  (`fpga/main_controller/stepper_motor_controller.v` → AXI4-Lite register wrapper, same
  packaging process already used for `fpga/camera_i2c/ov7670_axi_stream.v` → the
  `zynq_camera_sys` block design) is the established pattern for giving the PS software a
  register-level handle on new PL peripherals. A DAC-write/ADC-read peripheral for this bridge
  would likely follow the same AXI4-Lite convention rather than inventing a new bus protocol —
  noted as a reasonable default, not yet a decision.
- **Full-pipeline note (not this doc's main scope, but relevant context) — `PROPOSED`, not
  decided**: this project's motor-output convention already distinguishes RL-control-net
  output (raw step counts, zero conversion needed) from VLA-style model output
  (`theta_a/b/c` angles, needs `angle_to_steps()`). `CLAUDE.md` states the photonic arm runs
  "the same class of large VLA model as the Jetson arm," which would suggest its output is
  angles too — **if so**, the existing `angle_to_steps()` in
  `firmware/stm32_ml_control_and_vision/BallBalancingBot/MotorControl.h` should be reused for
  that downstream conversion rather than writing a new one, consistent with this project's
  reuse-over-duplication convention. **This is a conditional recommendation, not a confirmed
  fact**: it assumes Arm 3's actual model output format matches the Jetson VLA arm's
  `theta_a/b/c` convention exactly, which is not independently confirmed anywhere in evidence
  available to this agent (Arm 3's model/output format is itself downstream of the §1 framing
  question). Flagging the ownership now so this step doesn't get silently re-invented later,
  not asserting the format match is settled.

---

## 6. What this draft deliberately does not attempt

- No RTL, no pragma/IP-packaging detail, no AXI4-Lite register map — premature until §1 and
  §8 have real answers.
- No DAC/ADC part number recommendation — every placeholder above is explicitly flagged as a
  placeholder, not a choice.
- No laser-source plan — the paper's own Discussion flags this as unsolved even for its own
  real deployed-system case; doubly unresolved here given §1 is still open.

---

## 7. Fact ledger (what's confirmed-from-paper vs. assumption vs. unknown)

| Claim | Status | Basis |
|---|---|---|
| Compute core is passive; weights fixed at fab time | `CITED` (from paper, via notes §1) | Direct paper claim |
| 2N−1 DAC channels, C ADC/photodetector channels per inference | `CITED` (from paper, via notes §4) | Direct paper claim |
| ~11-bit DAC resolution | `DERIVED`, not a paper spec | Back-calculated from paper's 3mV/1.1-5V calibration sweep |
| Thermo-optic control too slow for real interfacing; EO/Pockels is the named upgrade | `CITED` (from paper, via notes §6) | Direct paper claim (Discussion) |
| Drift/recalibration behavior over time is uncharacterized | `CITED` (from paper, via notes §8) | Direct paper claim; leaves open whether a real bridge needs periodic re-calibration |
| Actual settling-time number, thermo-optic or EO path | `UNKNOWN` | Not given anywhere in source material |
| Photodetector/ADC part, bandwidth, resolution | `UNKNOWN` | Explicitly absent from paper (notes §8) |
| ZedBoard XADC header channel count/pinout | `UNKNOWN` | Not verified in this session; flagged, not asserted |
| Whether Arm 3 = real chip / sim-only / commercial product | `UNKNOWN` — blocking | Not answered anywhere in `CLAUDE.md`, logbook, or notes file |
| Does 120Hz/8.3ms loop bind Arm 3 | `UNKNOWN` | `CLAUDE.md` only carves out the Jetson arms explicitly |
| AXI4-Lite as the bridge's register convention | `PROPOSED`, not decided | Pattern-matched from Domain 4's existing porting precedent |
| Arm 3 output format matches Jetson VLA arm's `theta_a/b/c` convention, so `angle_to_steps()` applies downstream | `PROPOSED`, not decided | Inferred from `CLAUDE.md`'s "same class of large VLA model" language, not independently confirmed; downstream of the §1 framing question |

---

## 8. Open questions a real spec still needs answered

1. **Framing (§1)**: real fabricated chip, simulation/co-design-only, or commercial product?
   This gates almost everything else.
2. If (a): what N/C (input/output channel count) does this project's actual device have, or
   will have? The paper's N=10/15, C=6/10 are examples from someone else's fabrication, not a
   project spec.
3. What photodetector and ADC will actually sit at the C output ports? No candidate part has
   been evaluated.
4. What is the real settling time of whatever actuator this project ends up using (thermo-optic
   vs. an EO/Pockels-class modulator). This is the single biggest unknown for deciding whether
   an FPGA-speed bridge is even timing-feasible against this architecture.
5. Does the project's 120Hz/8.3ms control-loop requirement actually bind Arm 3, or is Arm 3
   evaluated on a different latency standard (check `docs/EVALUATION_STRATEGY.md`'s four
   metrics for whether latency is even one of them for this arm)?
6. Where does the laser source come from in a real deployed (non-benchtop) version, if (a)
   holds? Unsolved in the source paper itself.
7. Who owns/has access to the actual photonic hardware (university collaborator fabrication
   run? in-house? a vendor's product)? Determines whether any of this is buildable at all
   versus purely a simulation exercise.
8. Does the photonic core need periodic recalibration (re-exercising the monitoring taps in
   §4) during deployed operation, or does a single factory calibration hold for the device's
   useful life? The paper doesn't characterize drift, so this is unknown either way.
9. Does Arm 3's model actually output `theta_a/b/c` angles in the same convention as the
   Jetson VLA arm (§5's conditional recommendation to reuse `angle_to_steps()`), or does it
   have its own, different output head/format?

**Recommendation**: don't proceed to any RTL/HLS or AXI4-Lite register-map design for this
bridge until question 1 has a real answer from the user/PI — everything downstream of it
changes shape depending on the answer.

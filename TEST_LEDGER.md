# Test ledger

Every hypothesis test in the project, in the order it was run. Tests 1–32 are the original project ledger, numbered as recorded at the time. Later work is grouped into families (MI-33, CMI-34, MI-35, SBX) and the two momentum reference analyses (MOM-B, MOM-L) sit outside the numbered count. "Registration evidence" states what record shows the decision rule was fixed before results were seen; where that record is only notebook header text, it says so. Post hoc diagnostics are listed but are not counted as trials.

## Momentum (p1_momentum)

Format: ID | family | registration evidence | run date | sample | outcome | informed a later choice? | where reported

| ID | Family | Registration evidence | Run date | Sample | Outcome | Informed later choice? | Where reported |
|---|---|---|---|---|---|---|---|
| MOM-B | Momentum baseline (outside legacy count) | Phase B notebook cell B0 "settled parameters (frozen)" stated before the results; Drive created 2026-07-25 07:38Z; no commit history | 2026-07-27 (an earlier 2026-07-25 run also exists) | Full: rebalances 2011-01-31..2026-06-30, n = 179 (includes 2022+; pre-dates the hold-out rule) | Spread 1.740%/mo [0.769, 2.705]; Sharpe 0.847 [0.335, 1.503]; IC 0.0605 [0.035, 0.087] | Yes: frozen reference for M1–M7; its tercile split motivated M6 and MOM-L | Tables 1, 2; Figures 1, 3 |
| MOM-L | Liquidity-universe test (outside legacy count) | Notebook cell 0 "PRE-REGISTERED (locked BEFORE seeing results)": floors, headline, joint screen; Drive created 2026-08-05 05:17Z | 2026-08-05 | Full, n = 179 per floor; turnover primary, Amihud robustness | All 4 floors × 2 measures fail the joint screen (CI_low > 0.30); point Sharpe 0.39–0.76 | Yes: findings proposed a cost study (not run); post hoc retention framing added | Table 3; Figure 2 |
| 1–9 | M-batch / M1 tax and holding period (+M1b) | Notebook header: hypothesis, 6 + 3 sub-tests, DSR and hold-out bars fixed "before any result"; Drive created 2026-07-30 04:57Z | 2026-07-30 | Full 186 monthly marks; single look 131/55 | Not supported (no variant meets ≥90% preservation AND tax cut) | Yes: signal_prop weighting parked as a lead | Table 4 |
| 10–14 | M-batch / M3 volatility scaling | Findings file (hypothesis, test hierarchy); notebook created 2026-07-30 12:30Z; not opened in this revision | 2026-07-30/31 | Full 186; single look 2022–2026 | Not supported; tercile gap widened (point estimates) | Yes: motivated MOM-L ("tradeability gap") | Table 4 |
| 15–18 | M-batch / M4 reversal gate | Findings: "PRE-REGISTERED LOCK: raw gate @ N=21"; notebook created 2026-07-31 09:54Z | 2026-07-31 | Full, 179 months | Inconclusive under the registered t > 2 rule (t = 1.91) | No | Table 4 |
| 19–24 | M-batch / M5 trend age (return ×3, crash ×3) | Notebook cell 0 (headline proxy high52, estimation-with-CIs for crash leg); created 2026-08-02 09:23Z | 2026-08-02/03 | Full | Return leg inconclusive (CI includes 0); crash leg descriptive | Suggested a maturity-aware overlay (not run) | Table 4 |
| 25–28 | M-batch / M6 liquidity hump (Lind–Mehlum ×4) | Findings: test design and hump criteria; notebook created 2026-08-03 15:27Z | 2026-08-03 | Full, 198 rebalance months | Hump not supported | Yes: closed the "diffusion" lead; cited for the monotonic gradient | Table 4 |
| 29 | M-batch / M7 path entropy (gate step 1) | Findings: gated design, distinctness before performance; notebook created 2026-08-04 05:55Z | 2026-08-04 | Full (incl. 2022+) | Inconclusive; later gates not reached | Yes: led to the path-texture paper | Table 4; path-texture paper |
| — | Post hoc: 179 vs 186 reconciliation (revision diagnostic, not a trial) | n/a | 2026-10-14 | Archived counts | 7 extra M1 months = Phase B's 7 skipped months | No | Section 5 |
| — | Post hoc: retention ratios; bootstrap reproduction (diagnostics, not trials) | n/a | 2026-10-13 | Archived CSVs | Ratios 52–94%; CI reproduced exactly | No | Section 4.2; Section 3 |

## Path texture (p2_path_texture)

| ID | Family | Registration evidence | Run date | Sample | Outcome | Informed later choice? | Where reported |
|---|---|---|---|---|---|---|---|
| 29 (M7) | Legacy 1–32; M-batch internal count (M1:9, M3:5, M4:4, M5:6, M6:4, M7:1) | Gate chain and pass rules written in notebook markdown/code before the gate cell ran (cell exec 2026-08-04 06:35 UTC); no separate dated document | 2026-08-04 | Monthly 2011-01..2026-06 (incl. 2022+), 176 FM months | Gate 1 failed (t −0.12; uncontrolled −0.29); gates 2–4 not reached | Yes: Test D design (VR-first, the wavelet idea routed as a new hypothesis; D's gates 2–3 reference M7's 0.046 band and 0.44 stability) | §4, Table 2 |
| 30 (C gate) | Legacy 1–32; microstructure batch | Rule (≤0.20/≥0.40) stated in notebook before the gate cell ran (exec 06:15 UTC 2026-08-08) | 2026-08-08 | 2010-06..2021-12, 133 months | Ambiguous band (0.303) | Yes: triggered C diagnostics; C shelved before D | Appendix A |
| 31 (C diagnostics D1/D2) | Legacy 1–32 | Readings written in the same cell as the code; graded D2 bands replaced a binary 50% rule during the build; single recorded execution 06:28 UTC; treated as exploratory | 2026-08-08 | 132 months (liquid core) | Real, size-robust link (20% collapse); C shelved | C shelved | Appendix A |
| 32 (D gate) | Legacy 1–32; microstructure batch | Gate chain written in notebook before execution (07:48 UTC 2026-08-08) | 2026-08-08 | 2011-01..2021-12, 125 months (last forward window reads Jan 2022) | Gate 1 failed (t −0.23; uncontrolled −0.41); gates 2–3 not reached; wavelet build not run | Yes: "path-texture structural prior" recorded for future ideas | §4, Table 2 |
| REV-P2 (not a trial) | Revision diagnostics (post hoc) | None; added 2026-10-13/14 during drafting/revision | 2026-10-13/14 | as stated in §5 and Table 3 | M7 2011–21 re-estimate (t 1.09), D strict (t −0.28), tail-robust M7 gates, tie checks, nulls, intervals | No (descriptive) | §4–5, Table 3 |

## Mutual information (p3_mi_screen)
| ID | Family | Registration evidence | Run date | Sample | Outcome | Informed later choice? | Reported |
|---|---|---|---|---|---|---|---|
| MI-33 | MI screen, 30 cells | driver notebook states "characterization, not selection"; no winner ranked | 2026-08-10; calendar-audited rerun 2026-10-14 | 2010-01..2021-12 dev | original 30/30 MI>0 (Holm); audited 29/30 (overnight tilt 3m includes zero) | No; the four conditional pairs were fixed before the conditional run | Fig 1 |
| CMI-34 | 4 conditional pairs, 1m | pairs listed in notebook header before run | 2026-08-11 (naive), 2026-09-02 (corrected), reruns 2026-10-13 and 2026-10-14 (audited calendar) | dev | mom/vol net redundancy; rev/mom weakly redundant on audited calendar (CI upper −0.00002; inconclusive before); ami/turn inconclusive | no | Table 2 |
| MI-35 | rank IC + extremeness MI | rule fixed in rerun notebook header before real-data stage | 2026-10-13; audited 2026-10-14 | dev | audited: 10 direction-led, 6 extremeness-led, 13 mixed, 1 no signal (9 labels changed vs pre-audit) | no | Fig 1–2 |
| — | Holm, block sensitivity, sign shares | post hoc revision diagnostics | 2026-10-14 | dev | see text | no | §4–5 |

## Models vs rules (p4_models_vs_rules)

The family is outside the legacy 32-test ledger, per the editor's convention. Registration evidence for every row: a pre-committed rule in the notebook's header markdown cell (before the results cells); no separate timestamped record (see BLOCKERS B9). Sample: V1 dataset; test years 2014–2025 (includes 2022–2025; training 2011–2024).

| ID | Family | Test | Registration evidence | Run date | Sample | Outcome | Informed later choice? | Where reported |
|---|---|---|---|---|---|---|---|---|
| SBX-2.1 | SBX / Phase 2 | Outperform: M1 beats M0 (rule 1) | phase2 notebook Cell 0 | 2026-09-30 | 1,411,188 test cards | pass (skill 0.0027 [0.0002, 0.0052], i.i.d. CI; block rerun pending) | yes: M1 outperform → Phase 3a pick portfolio | Table 1 |
| SBX-2.2 | SBX / Phase 2 | Outperform: M2 materially beats M1 (rule 2) | same | 2026-09-30 | same | fail (gain −0.0001) | yes: M1 retained | Table 1 |
| SBX-2.3 | SBX / Phase 2 | Top performer: rule 1 | same | 2026-09-30 | same | pass (0.0022 [0.0006, 0.0037]; block rerun pending) | yes: Phase 3a | Table 1 |
| SBX-2.4 | SBX / Phase 2 | Top performer: rule 2 | same | 2026-09-30 | same | fail | yes: M1 retained | Table 1 |
| SBX-2.5 | SBX / Phase 2 | Crash: rule 1 | same | 2026-09-30 | same | pass (0.0192 [0.0134, 0.0245]) | yes: Phase 3a | Table 1 |
| SBX-2.6 | SBX / Phase 2 | Crash: rule 2 | same | 2026-09-30 | same | fail (gain 0.0013 < 0.002 materiality; CI > 0) | yes: M1 retained | Table 1 |
| SBX-2.7 | SBX / Phase 2 | High vol: rule 1 | same | 2026-09-30 | 1,411,156 test cards | pass (0.1292 [0.1205, 0.1378]) | yes | Table 1 |
| SBX-2.8 | SBX / Phase 2 | High vol: rule 2 | same | 2026-09-30 | same | pass (gain 0.0143 [0.0117, 0.0170]) | yes: M2 used for avoid-wild-movers portfolio | Table 1 |
| SBX-3a.1 | SBX / Phase 3a | Pick outperformers adds return | phase3a_v2 Cell 0 | 2026-10-04 | 141 months 2014–2025 | fail (active +3.72% [−1.60, +8.67]) | yes: tilts → comparator design | §4.2, Table 3 |
| SBX-3a.2 | SBX / Phase 3a | Pick big winners adds return | same | 2026-10-04 | same | fail (−0.60% [−4.69, +3.08]) | no | §4.2 |
| SBX-3a.3 | SBX / Phase 3a | Avoid crashes reduces risk / worth it | same | 2026-10-04 | same | pass / pass (Sharpe diff [+0.085, +0.258]; holds at 98.75%) | yes: comparator design | §4.2, Fig. 2 |
| SBX-3a.4 | SBX / Phase 3a | Avoid wild movers reduces risk / worth it | same | 2026-10-04 | same | pass / pass ([+0.075, +0.237]; holds at 98.75%) | yes: comparator design | §4.2, Fig. 2 |
| SBX-3b.1 | SBX / Phase 3b | Avoid crashes vs cut-most-volatile | phase3b Cell 0 | 2026-10-05 | same | "model adds value" at 95% (+0.025 [+0.003, +0.060]); not at 98.33% ([−0.001, +0.068]) | no (follow-ups proposed, not run) | §4.3, Fig. 2 |
| SBX-3b.2 | SBX / Phase 3b | Avoid wild movers vs cut-most-volatile | same | 2026-10-05 | same | no detectable difference (+0.011 [−0.008, +0.041]) | no | §4.3 |
| SBX-3b.3 | SBX / Phase 3b | Pick outperformers vs momentum−volatility (hindsight comparator) | same | 2026-10-05 | same | comparator better (−0.119 [−0.244, −0.040]; 98.33% [−0.283, −0.019]) | no | §4.3, Fig. 2 |

Not in the family (diagnostics, labelled in the paper): the momentum-only reference pair; comparators vs baseline; the linear-regression diagnostics (v1_regression, v2_residualized); the conditional-probability persistence check; the step-2 Markov persistence figure; and every revision-round diagnostic (approximate vol-baseline AUC, the pending block rerun).

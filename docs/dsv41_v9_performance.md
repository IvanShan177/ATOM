# DSV4.1 V9: locked current-main performance comparison

The retained V9 combination reduced full C256 service time in both paired runs
against clean main `43eed2fde149e0d2e8986964af46fe1a02ff52ca`. Pooled output
throughput increased from **5184.803 to 6982.727 tok/s: +34.6768%, 1.346768×**.
This is a directly measured total-version comparison, including upstream
scheduler/attention differences. It is not an isolated kernel attribution.

| Pair | Main output tok/s | V9 output tok/s | Change |
|---|---:|---:|---:|
| A1/B1 | 5169.961 | 7007.291 | +35.5386% |
| A2/B2 | 5199.731 | 6958.334 | +33.8210% |
| Pooled | 5184.803 | 6982.727 | +34.6768% |

Main repeat drift was +0.5758%; V9 drift was −0.6987%. All 10240 formal requests
succeeded. TTFT, TPOT, ITL and E2EL p99 improved in both pairs. Formal AL values
were 2.464579/2.502186/2.479151/2.473393 for A1/B1/A2/B2 respectively. Each arm
generated 2354462 output tokens; pooled throughput uses total tokens / total time.

The candidate used branch HEAD `0653b59a75930397a487654aca753bada461af35`, retained
P1 dirty diff `10bd70546a3850fea421e794abae88e1a61362391cb5da89064fd87893dbb7db`,
and frozen V9 runtime overrides now preserved in `tools/dsv41/v9`. AITER was
`c141da6bd10300dfc22da7777c23efc6c7b88712`; model was DeepSeek-V4.1-Flash.
Common settings: physical GPUs4–7, C256, random8192 input/1024 output,
range0.8/seed0, DSpark5, TP4/EP1/DP1, FP8 KV/FP4 index, FULL level3,
maxseq512/maxbatched16384/chunk16384, GPU utilization0.9. Each arm ran
512 regular + C8/C12 supplemental warmup (20 requests), then 2560 formal requests.
Formal request fingerprints, client source, resolved external configuration,
model metadata and token counts matched. B1/B2 had zero new owner captures
in the formal interval on all four ranks.

The earlier C8-only main A1 is diagnostic only. Its paired candidate failed
before formal timing because M32 coverage was missing. Both original runs are
retained; all four reported V2 arms use the same C8+C12 protocol.

The runtime reuses one owner quantization and single AGV for FP8 values, E8M0
scales and router logits, preserving shared/routed consumers. Ordered BF16 RS
preserves the calibrated native summation order. Prefill uses explicit raw
top6 routes and fused route reduction/shared join/push RS with the producer
system fence. G1 wait0 corrections retain their tested owner/M4096 scopes.
A0/H64 attention was already committed in `8ba40a934`. New projection Quant/Sum
prototypes are excluded. All paths remain opt-in and preserve guarded fallback.

Qualification limits:

- Task accuracy/GSM8K was not run. Local numerical and successful service
  requests do not establish task accuracy.
- Candidate-only background owner observation has unqualified transfer overhead.
  The formal benchmark timer/math is unchanged; boundary snapshots are outside
  that timer. Removing observers is a new measurement protocol.
- Main formal cold-compilation absence was not independently qualified.
- Hardware sampling used approximate formal epoch brackets. Four-rank GFX
  medians were 2314–2342MHz, hotspot medians53–59°C, with zero sampler errors;
  throttle status was N/A. These observations do not prove causal attribution.
- The measured reduction calibration and binary/compiler/dependency contracts
  are specific to this deployment. Rebuilds and other machines need qualification.

The previous **same-version H64 branch increment +2.817003%** remains a separate
experiment, below its +3% minimum target. Historical +19.19% is also separate.
Neither is multiplied into the current-main total ratio. Service-level individual
owner/prefill/attention contributions and new interaction terms were not measured
here; retain the prior local four-cell/best-single/parent records separately.

Authoritative external evidence:
`validation/mainline_abab_20261031/round_main43_v9_v2/{RESULTS.md,SUMMARY.json,RUNTIME_AUDIT.json}`
and the four `main43_v9p2_*` frozen source, client/server, count and lifecycle runs.
Registry verification covered 54 candidates,459 archived phases,14222 evidence
hashes with no errors; five early historical indexes remain partial backfills.
The original P1 and AITER CSVs were preserved. This commit preserves the measured
source and reproduction conditions; it does not claim a new post-commit GPU run
or complete optimization/accuracy acceptance.

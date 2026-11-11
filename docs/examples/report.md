<!-- Demo: scripted model output on synthetic data; not a real LLM evaluation. -->

# Investigation demo — scripted model output

Synthetic telemetry with artificial labels. These are diagnostic leads, not confirmed faults.

## Triaged cycles

| Cycle | Model p_degraded | Trend | Example tool evidence |
|---|---:|---|---|
| synthetic-000002 | 0.999 | persistent | volume_flow_1_std = 0.122; healthy mean = 0.15 |
| synthetic-000006 | 0.999 | persistent | volume_flow_1_std = 0.117; healthy mean = 0.15 |
| synthetic-000010 | 0.999 | persistent | pressure_1_std = 0.1; healthy mean = 0.081 |

## Proposed diagnostic rule

pressure_1_mean > 10.0: measured precision 0.75, recall 1.0 on the synthetic labeled table. Human approval controls saving the proposal.

## Next checks

Review the flagged cycles in the original signals, inspect adjacent cycles, and test the rule on independent labeled data before use.
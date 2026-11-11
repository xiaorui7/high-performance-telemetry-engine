# Experimental anomaly investigation
Optional dependencies: install with `python -m pip install -e ".[ml,agent]"`.
Install the chosen provider separately, for example `langchain-openai` or
`langchain-anthropic`, and configure its API key. Set TELEMETRY_AGENT_MODEL
to a provider-qualified model identifier supported by init_chat_model.

## Flow
triage -> investigate -> propose_rule -> human_approval -> write_report -> verify
Verification retries report generation once, then emits a warning if still ungrounded.
LangGraph InMemorySaver and interrupt/Command preserve state during approval.

## Deterministic tools
- triage ranks model degradation probability, then flag count for ties.
- compare_to_healthy uses healthy-training mean/std. Constant features are marked
  explicitly rather than dividing by zero.
- neighbor_trend uses CSV row order as chronology; it reports persistent risk when
  at least half the surrounding cycles cross the documented probability threshold.
- test_rule computes descriptive precision/recall on the supplied labeled table.
  It is not a held-out rule-selection study.

The model chooses between comparison and trend tools within a bounded per-cycle
budget. Prompts v1/v2 are selectable; v2 requires neighbor evidence for degraded calls.
Rules are saved only after approval, exclusively to rules_proposal.yaml next to the
report, using exclusive creation. Existing configurations are never edited.

## Run
```bash
telemetry investigate --results models/valve --model models/valve --table models/valve/table.csv --report results/investigation/report.md --trace results/investigation/trace.json --prompt-version v2
python scripts/eval_agent.py --results models/valve --model models/valve --table models/valve/table.csv --confirm
```
The CLI shows proposed rule evidence and requests y/n. --yes explicitly opts into
automatic approval for scripts. The evaluation runner declines proposed rules.
Traces record model messages, tool inputs/outputs, available token usage, and elapsed
time. Do not send private telemetry to a hosted provider without authorization.

## Evaluation and limitations
No real-provider evaluation has been recorded yet. Offline tests use scripted fake
chat responses, never an API key. Run the evaluation to create evals/agent_report.json;
do not present fake responses as provider-quality evidence.
Numeric/cycle-ID verification is a lexical grounding guard, not a semantic proof:
it cannot establish causality, detect all misleading prose, or guarantee that a
number is attached to the correct feature. Prompt adherence is not guaranteed.
Probability is model risk, not calibrated fault probability. Evaluation defaults to blocked-CV out-of-fold predictions on public UCI data.
Healthy reference statistics still use all-training samples, and proposed-rule scores
remain descriptive full-table scores; the whole agent is not a fully nested held-out
evaluation. Explicit model-source evaluation uses in-sample predictions. Report degradation calls are structured
alongside prose; contradictions in prose are not completely detected.
The memory checkpointer lasts only for the current process. Invalid model JSON
returns an explicit error rather than an unbounded retry.

## Next steps for the team
FastAPI service; Docker; deployment and secrets management; LLM-as-judge scoring;
trajectory evaluation; multi-agent designs; web UI; persistent checkpoint storage;
monitoring and alerts.

## OOF evaluation and baseline
Re-run telemetry train to produce oof_predictions.csv. Triage and neighbor_trend
accept model (default for investigate) or oof; evaluation defaults to oof.
The seeded sampler selects disjoint contiguous blocks in cycle_index order; incomplete
trailing blocks are excluded. Both prompts and ranking baseline see identical blocks.
The baseline calls the highest OOF-risk cycles degraded using the agent's investigation
budget. Recall uses all degraded cycles in each slice, not just investigated cycles.
Precision/recall are pooled across slices; no positive calls gives null precision.
Grounding is the existing lexical guard; structured degraded_cycles supplies the
classification decision. Ungrounded runs remain included and are flagged.
Tokens are null when any provider usage is unavailable, not guessed as zero.
Baseline grounding and runtime are N/A; it has no LLM calls.
The CLI prints an upper bound on LLM calls and requires --confirm before any calls.
Override --slices, --slice-size, --seed, or --top-n as needed.

## Offline demo and compiled graph
Install optional extras, then run:
```bash
telemetry investigate --demo --yes --report results/demo/report.md --trace results/demo/trace.json
```
Omit --yes to inspect and approve/decline the proposed rule interactively.
The demo trains on temporary synthetic signals with artificial labels and uses the
same scripted fake chat factory as demo tests. It is not a real-provider evaluation.
Use a fresh report directory when approving, because proposals are never overwritten.
See [agent_graph.md](agent_graph.md) for the actual compiled Mermaid graph.

<!-- evaluation-results -->

## Evaluation results

Full default evaluation: not completed. OpenAI returned a requests-per-day rate limit.
The following is the single-slice pilot, not the default evaluation.
Model: gpt-4.1-mini-2025-04-14. Source: evals/agent_pilot_fixed.json.

| Method | Precision | Recall | Grounding | Avg tools | Avg seconds | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 1.0 | 0.04285714285714286 | 0.0 | 8.0 | 10.76294990000315 | 15919 |
| v2 | 1.0 | 0.04285714285714286 | 0.0 | 8.0 | 41.18427770002745 | 16248 |
| ranking_baseline | 1.0 | 0.04285714285714286 | N/A | 0.0 | N/A | 0 |

Both prompts matched ranking precision/recall in the pilot but failed lexical grounding. Rounded numbers and unsupported prose remain visible in retained reports. No benefit over ranking is established. Invalid rules are rejected without approval; failed runs retain traces. Usage accounting is in evals/agent_usage.json; an earlier failed pilot lacked a retained trace.

<!-- uncertain-evaluation-results -->

## Uncertain-cycle evaluation

Status: **partial**. Source: `evals/agent_uncertain_report.json`. OOF probabilities on public UCI cycles; the health statistics still come from the full training table.

| Method | N | Precision | Recall | F1 | Accuracy | Grounding | Failure | Calls | Tokens | Est. USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_a | 3 | 0 | 0 | 0 | 0.6667 | N/A | N/A | 0 | 0 | 0 |
| baseline_b | 3 | 0 | 0 | 0 | 1 | N/A | N/A | 0 | 0 | 0 |
| v1 | 1 | 0 | 0 | 0 | 0 | 0.5 | 0.5 | 1 | 825 | 0.0003984 |

Partial data cannot establish comparative performance. Missing agent decisions are excluded, not imputed.

# Main-Materials GPT-5.5 Direct-Judge Subset

This run evaluates a one-shot LLM-as-judge baseline on the same validation/test subset used by local VoI and Sci-VoI-RHI.

- Records: `474`.
- Validation/test: `235` / `239`.
- Best primary utility: `local_voi_harness`.
- Best risk-adjusted utility: `local_voi_harness`.

## Results

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Coverage | Utility efficiency | Diagnostic score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_heuristic` | 0.3067 | 0.1022 | 0.6667 | 0.3043 | 0.0628 | 0.3067 | 0.7444 |
| `llm_direct_judge` | 0.0712 | 0.0465 | 0.3462 | 0.6522 | 0.1088 | 0.0712 | 0.3152 |
| `local_voi_harness` | 0.8233 | 0.8233 | 0.0000 | 0.2609 | 0.0000 | 0.0509 | 0.3324 |
| `scivoi_rhi` | 0.0207 | 0.0103 | 0.5000 | 0.5217 | 0.0335 | 0.0509 | 0.3324 |
| `verbal_confidence` | 0.0325 | 0.0101 | 0.6906 | 0.5217 | 0.5816 | 0.0325 | 0.4324 |

## Protocol

- `llm_direct_judge` sees only visible action context, candidate action, and pre-execution evidence; it outputs `p_success` once per action.
- `local_voi_harness` trains a non-recursive local value-of-information harness on the validation partition.
- `scivoi_rhi` uses train/feedback/acceptance partitions from validation data to recursively mutate and accept the harness before test evaluation.
- The held-out test split is used only for final reporting.

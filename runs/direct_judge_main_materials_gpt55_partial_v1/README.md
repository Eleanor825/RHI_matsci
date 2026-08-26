# Main-Materials GPT-5.5 Partial Evaluation

This is an auditable partial evaluation on the same 500-record main-materials
subset prepared for the direct-judge experiment. The historical `gpt-5.5`
cache contains 474 of the 500 records; the remaining 26 records could not be
rescored on August 20, 2026 because the current `mimorouter` deployment has
no `gpt-5.5` channel and the historical `hi-code` endpoint rejected the
current key.

## Provenance

- Source records: 500.
- Scored records: 474 (`235` validation, `239` test).
- Missing records: 26.
- Judge model: `gpt-5.5`.
- Historical score provider: `hi-code`.
- Current provider check: `mimorouter` lists only `gpt-5.6-luna`,
  `gpt-5.6-sol`, and `gpt-5.6-terra`; requesting `gpt-5.5` returns HTTP 503
  with `model_not_found`.
- The partial result must not be reported as a complete 500-record
  main-materials comparison.

## Same-subset Evaluation

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Coverage | Utility efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local_voi_harness` | 0.8233 | 0.8233 | 0.0000 | 0.2609 | 0.0000 | 0.0509 |
| `llm_direct_judge` | 0.0712 | 0.0465 | 0.3462 | 0.6522 | 0.1088 | 0.0712 |
| `evidence_heuristic` | 0.3067 | 0.1022 | 0.6667 | 0.3043 | 0.0628 | 0.3067 |
| `verbal_confidence` | 0.0325 | 0.0101 | 0.6906 | 0.5217 | 0.5816 | 0.0325 |
| `scivoi_rhi` | 0.0207 | 0.0103 | 0.5000 | 0.5217 | 0.0335 | 0.0509 |

The `local_voi_harness` row is conservative on this partial split and has
zero threshold-selected coverage; its top-budget utility and threshold
coverage should therefore not be interpreted as the same operating point.
The complete main-materials Sci-VoI-RHI result remains the authoritative
method result in `runs/scivoi_rhi_main_materials_v1/`.

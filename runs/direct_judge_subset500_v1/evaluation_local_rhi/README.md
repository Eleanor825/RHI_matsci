# GPT-5.5 Subset: Local Harness Comparison

This evaluation uses the complete 500-record historical GPT-5.5 subset and
the same validation/test split as the cached direct judge (`249` validation,
`251` test). It evaluates the local VoI harness and Sci-VoI-RHI without new
LLM calls, using the existing `gpt-5.5` score cache for the direct-judge row.

| Method | Net utility | Risk-adjusted | Risk | Hit rate | Coverage | Utility efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local_voi_harness` | 0.8357 | 0.5571 | 0.3333 | 0.2000 | 0.0120 | 0.1066 |
| `scivoi_rhi` | 0.0861 | 0.0359 | 0.5833 | 0.4000 | 0.0956 | 0.1066 |
| `llm_direct_judge` | 0.0558 | 0.0338 | 0.3939 | 0.6800 | 0.1315 | 0.0558 |

The proxy utility and binary label are not identical targets in this older
subset. The local VoI row is also highly conservative, with only `1.2%`
threshold-selected coverage. Therefore the result supports a trade-off claim,
not a claim that the local method uniformly beats GPT-5.5.

# Same-Evidence Numerical Comparison Note

Recorded on August 23, 2026, before computing any LLM-judge metric.

The outcome-blind LLM subset contains 485 acceptance records: all 85
`discover_unique` records and 200 records each from `extreme_properties` and
`matbench_pairwise`. Comparing its LLM gain to the numerical harness gain on the
complete 2,538-record acceptance batch would be invalid because both the 10%
execution budget and the evidence-allocation portfolio depend on batch
composition.

The frozen H5 runner was therefore rerun on exactly the same 485 acceptance
records, with all original 14,630 training and 3,601 feedback records unchanged.
No LLM prediction or LLM metric had been inspected. The subset artifact has SHA
`d85a27e458d061b242a95d3a11fe3c1cf2d4b936aa9106ec2069dd37e24b1191`.

On this exact subset:

- learned portfolio KG strict trajectory gain: 32.2719;
- base-only factorized LCB strict trajectory gain: 33.4955;
- learned portfolio KG selective risk: 2.04%, 95% UCB 9.32%;
- base-only selective risk: 2.04%, 95% UCB 9.32%.

The prespecified LLM gate continues to compare the LLM judge with the learned
portfolio-KG primary method. Base-only is also reported as the strongest
same-subset numerical comparator. The primary is not redefined after observing
this result.

# Pre-Inference Protocol Amendment

Recorded on August 23, 2026, before any strict-prompt model inference or
inspection of expanded acceptance/test outcomes.

- Model reasoning effort is fixed to `medium` for all three GPT-5.6 replicas.
- The acceptance-only runner excludes all test trajectories before loading LLM
  caches or constructing factorized states.
- Test caches and test evaluation will be created only if the primary
  acceptance certificate at `alpha=0.10`, `delta=0.05` succeeds.
- Expanded deterministic trajectory SHA-256:
  `746b312fc40e5a3f99ba5423dd87d1319ca38d33bc3a3a1fab87752ff558fd28`.
- Expanded cross-fit trajectory SHA-256:
  `5ce98d5960d6d5ad54dca0be2e4106d559cc8e6112dcc624c0c90fbfc52cee43`.
- Final acceptance/full runner SHA-256:
  `88664b3e5fa53f4c340ae1350ebaa73471ee5770c768a042115de0df13d6397e`.

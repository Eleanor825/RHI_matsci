# Prompt-Conditioned Materials Trajectories v1

This run contains 900 prompt-conditioned trajectories: 300 instances from each of the three main materials tasks. Each trajectory has at most 3 steps and records the full system prompt, user prompt, available action menu, agent output, internal/external signals, route, and hidden benchmark outcome.

This first run uses a deterministic prompt policy (`agent_backend=deterministic_prompt_policy`) so the trajectory construction is reproducible without an API key. `prompt_generated=true` means the action is selected from a prompt-defined action menu; `llm_generated=false` is explicit. The same runner can later replace the policy with an LLM backend without changing the trajectory schema.

The prompt never includes label, utility, post-action reward, or raw oracle fields. Outcomes are retained only for evaluation.

Task counts: `{"discover_unique": 300, "extreme_properties": 300, "matbench_pairwise": 300}`.

The deterministic policy is a construction baseline, not the proposed harness. Replay the same prompts with no-harness, static-gate, and Sci-VoI-RHI policies before drawing method conclusions.

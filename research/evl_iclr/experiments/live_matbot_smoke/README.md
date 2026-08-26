# Live MatBot Runtime Evidence

## What Was Run

- OpenClaw `2026.5.18` executed the real high-energy-density-battery MatBot
  workspace using file-reading tools and the active NCM request.
- Every successful run received the same prompt and had no access to hidden
  experimental outcomes.
- The agent returned one proposed experiment plus seven numeric fields. It did
  not return a route label; routing remains a downstream policy decision.

## Successful Captures

| model | independent runs | tool calls | tool errors | post-outcome |
|---|---:|---:|---:|---:|
| GPT-5.6 Sol | 1 | 29 | 7 | no |
| GPT-5.6 Luna | 2 | 30 | 4 | no |

The three imported trajectories and their hashes are listed in
`manifest.json`. Provider-unavailable, model-not-found, timeout, and runtime
authentication failures are also preserved there rather than omitted.

## Main Diagnostic Result

The two independent GPT-5.6 Luna runs reported almost identical internal
uncertainty (`0.43` and `0.42`; range `0.01`) and identical evidence value
(`0.94`). Nevertheless:

- action token Jaccard was only `0.1791`;
- one run proposed a four-arm LiPO2F2/LiODFB factorial;
- the other proposed a single baseline anchor experiment;
- the derived pre-outcome net-gain proxy differed by `0.2698`.

This is evidence that a verbal/numeric self-report is not a sufficient measure
of internal action uncertainty. Internal uncertainty must be estimated from the
distribution of actions or action values under repeated sampling and controlled
perturbations, then calibrated against observed outcomes.

## Claim Boundary

These artifacts establish real runtime capture and expose same-model action
instability. They do **not** establish improved scientific utility because none
of the proposed experiments has a linked post-outcome event. The derived net
gain remains a pre-outcome proxy, not realized scientific gain.

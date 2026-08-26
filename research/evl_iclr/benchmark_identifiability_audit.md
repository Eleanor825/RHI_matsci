# Benchmark Identifiability Audit

## Finding

The current historical conversion is deliberately leakage-resistant but over-sanitizes the scientific state. It removes not only post-execution outcomes, but also task-defining pre-execution database/simulation outputs. Consequently, the action-worthiness target is often not identifiable from the information shown to the agent.

## Task Audit

### Matbench Pairwise

- Runtime-visible: composition, structure class, two noisy surrogate scores, surrogate margin, and estimated noise.
- Hidden evaluation: measured/computed Materials Project `log10(K_VRH)` and the true pairwise gap.
- Assessment: identifiable only to the quality of the supplied surrogate. A higher-fidelity property-prediction or database-query tool is a legitimate additional action, but the hidden true gap must not appear in the base prompt.

### DiSCoVeR-Style Unique Materials

The raw benchmark builder supplies:

- `log10(K_VRH)` from Matbench/Materials Project;
- a normalized performance score;
- a composition/structure uniqueness score;
- their combined discovery score.

The historical converter removes all three scores and rewrites the property evidence as the literal string `property estimate`. The remaining composition and space group are insufficient to reproduce a top-decile performance–uniqueness decision.

Permissible v2 tool evidence:

- database/property-model estimate of `log10(K_VRH)` with provenance and uncertainty;
- composition/structure novelty computed against a train-only reference library;
- independent model replicas or retrieval databases.

The exact combined discovery score and top-decile label remain hidden. If the database value itself defines the benchmark outcome, it must be exposed only as an explicitly costly high-fidelity query and the benchmark must be described as tool-acquisition rather than experimental prediction.

### RL-CC Extreme Properties

The raw benchmark builder contains the generated molecule's MW, logP, TPSA, QED, HBA, HBD, and DRD2 values, target values, per-property errors, and published tolerance bounds. These are outputs of computational property evaluation, not prose-only reasoning targets. The historical converter removes them and shows only the SMILES and target identifier.

Permissible v2 tool evidence:

- property-prediction/simulation outputs for the seven properties;
- per-property predictive uncertainty;
- independent model or conformer replicas;
- the published target and tolerance bounds.

The exact `all_hit` result remains hidden until all tool predictions are frozen. If exact generated properties are replayed from the paper, they are a high-fidelity simulation tool call with explicit cost, not base-state evidence.

## V2 Trajectory Contract

Each action becomes a partially observed trajectory:

```text
base scientific state
  -> scalar worthiness estimate
  -> optional database/retrieval/simulation tool call
  -> updated evidence view
  -> updated scalar worthiness estimate
  -> host policy executes or abstains
  -> hidden outcome revealed for evaluation only
```

The uncertainty harness never outputs a route. It outputs scalar positive-gain probability and source variances. The host MatBot policy decides whether the value of another tool call exceeds its cost.

## Required Leakage Controls

1. Split groups before fitting property models, novelty references, calibrators, or noise scales.
2. Tool observations for a target may use only train-library outcomes or independently executed models.
3. Exact target labels, combined benchmark scores, and post-execution rewards never appear in prompts.
4. High-fidelity replayed paper outputs are labeled as replay tools and charged a cost.
5. Results must separately report base-only, low-fidelity-tool, and high-fidelity-tool settings.
6. The paper must distinguish controlled/semi-synthetic observation channels from live MatBot tools.

## Consequence for Existing Results

Zero strict coverage on the sanitized benchmark is evidence that the current observation contract is under-informative, not evidence that uncertainty estimation is impossible. Conversely, restoring exact outcome-derived fields to the base prompt would create trivial leakage. The valid next experiment is costed sequential evidence acquisition with auditable tool provenance.

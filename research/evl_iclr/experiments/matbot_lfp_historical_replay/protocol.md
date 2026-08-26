# MatBot LFP Historical Replay Protocol

## Status

This is a **development-only retrospective replay**. The source table was
inspected before the protocol was written, so the result cannot be called
untouched confirmation.

## Unit of Analysis

Each record is one historical LFP A/B full-cell formulation pair that was
actually tested. The decision-time record exposes formulation, test protocol,
and provenance but withholds all measured performance values.

The release also contains one complete one-step trajectory per record. Its
`pre_action` event contains no uncertainty estimate and no outcome; its linked
`post_outcome` event contains the real measurement-derived utility and marks the
trajectory terminal. These trajectories can be replayed through a candidate
harness without presenting the hidden result to the agent.

## Outcome and Utility

- Legacy batches use converted -20 C DCR and 45 C cycle-600 retention.
- The ABA batch uses -15 C DCR at 50% SOC, 1 C, 10 s.
- Lower DCR and higher cycle retention are converted to empirical ranks within
  the same measurement protocol.
- Scientific utility is the mean of the available within-protocol rank scores.
- The binary action label is one only when utility is above the batch median.

This rank definition avoids comparing incompatible absolute DCR protocols, but
the dataset remains small and observational. It is intended to validate the
real-outcome replay pipeline and generate future MatBot trajectory tasks, not
to establish the primary paper claim.

## Pairwise Action-Choice Instances

Within each measurement protocol, every unordered pair of tested formulations
forms one retrospective choice state. This produces 19 A/B instances from the
12 experiments. Candidate orientation is determined by a stable hash of record
IDs and never uses performance outcomes. The agent sees both formulations and
must choose which action to execute; evaluation reveals both measured utilities
and checks whether the selected action was superior.

These instances are designed for repeated multi-model sampling. Internal action
uncertainty can be measured from empirical A/B choice distributions, while
external uncertainty can be measured by rerunning the same state under frozen
evidence interventions. The harness still emits numeric worthiness rather than
a route label.

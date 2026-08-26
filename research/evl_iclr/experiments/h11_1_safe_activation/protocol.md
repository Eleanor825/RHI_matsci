# H11.1 Protocol: Safe Activation of a Frozen Acquisition Contract

Date locked: 2026-08-21

The task-specific acquisition thresholds are fitted and frozen on feedback as
in H10. The independent acceptance split evaluates exactly one nontrivial
contract with the original eight-point exact-risk grid at `delta=0.05`.

If the frozen contract has nonzero certified coverage and positive acceptance
population net gain after tool costs, activate it. Otherwise activate no-op,
which acquires no tool evidence and executes no action. Since there is only one
risky contract and no-op has risk zero, no additional acquisition-policy
multiplicity correction is introduced.

Seed 7 must preserve positive H10 test gain. The five-seed target is
non-negative gain in every seed, positive gain and nonzero coverage in at least
four seeds, and empirical risk at most 0.20 whenever active.

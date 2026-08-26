# Pre-evaluation clarification

Recorded on August 23, 2026, after trajectory export but before building H5
features or running any round-2 acceptance/test metric.

The round-2 train partition may be used only to fit the controlled benchmark's
pre-acquisition tool-prior environment and its cross-fitted replicas. This is
part of constructing visible observations. It is not used to fit the action-
worthiness model, gain calibration, portfolio-KG residual head, acquisition
policy, or execution policy.

All harness fitting, calibration, and policy feedback remain restricted to the
round-1 train and feedback partitions. Round-2 acceptance and test outcomes are
never used for fitting. The round-2 test remains physically separated from the
acceptance runner.

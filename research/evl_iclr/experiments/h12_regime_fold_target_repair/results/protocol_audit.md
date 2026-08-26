# H12 Protocol Audit

The deterministic five-fold assignment was audited before running scientific
tools.

| task | benchmark groups | groups covered exactly once | test overlap | reservation range across folds |
|---|---:|---|---|---:|
| Matbench pairwise | 28 | yes | none | 0.00002241 |
| unique discovery | 7 | yes | none | 0.00001499 |
| extreme properties | 10 | yes | none | 0.00001307 |

The legacy repeated-seed protocol held out the same pairwise and
unique-discovery regimes in every run. H12 instead changes the scientific test
regimes in every fold while covering the complete regime set exactly once.

The repaired `potential_success_slot_value` reservation remains approximately
0.8699, 0.8250, and 0.7797 for the three tasks in every fold. It removes the
legacy discontinuity where the unique-discovery reservation became zero when
training success prevalence fell slightly below the 10% action budget.

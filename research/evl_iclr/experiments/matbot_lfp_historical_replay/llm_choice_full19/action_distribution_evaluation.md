# Frozen action-distribution evaluation

- Protocol: leave one historical source batch out; 19 pairwise actions from
  three real LFP experiment batches.
- Sampling: GPT-5.6 Sol, Luna, and Terra; three replicas per model and record;
  171 successful model calls with no missing cells.
- Raw multi-model direct probability: accuracy `0.8421`, Brier `0.19017`, log
  loss `0.56970`, AURC `0.30734`.
- Self-reported-uncertainty shrinkage: accuracy `0.4211`, Brier `0.27042`, log
  loss `0.73498`, AURC `0.56905`.
- Action-entropy shrinkage: accuracy `0.4211`, Brier `0.27136`, log loss
  `0.73694`, AURC `0.57091`.
- Action entropy plus between-model variance: accuracy `0.4211`, Brier
  `0.27136`, log loss `0.73693`, AURC `0.57091`.
- Action entropy was nonzero on `42.1%` of actions, but its correlation with
  raw direct-probability error was only `0.0117`.

## Decision

The preregistered success condition was not met. Directly shrinking an LLM
probability using empirical action entropy is rejected. This result is
development-only because the historical source records were inspected before
protocol lock. The replacement method treats action variation as a crossed
internal/external distribution to be decomposed and calibrated, not as a
monotone uncertainty penalty.

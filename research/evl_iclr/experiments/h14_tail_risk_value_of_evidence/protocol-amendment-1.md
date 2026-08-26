# H14 Protocol Amendment 1

Date: 2026-08-22

The original protocol used a difference between two predicted tail-risk scores
as the realized value-of-evidence target. Before reading any external target
distribution or running any H14 model on the external datasets, this was
replaced by the realized counterfactual policy value

```text
Delta_t = G * (D(q_t) - D(q_base)) - c_t.
```

This clarification aligns acquisition training with the declared optimization
target, net scientific gain. It does not alter datasets, folds, baselines,
alpha, delta, coverage candidates, tool costs, or confirmatory gates. The v1
hash is retained in `protocol-lock-v1.sha256`; `protocol-lock.sha256` contains
the amended protocol hash.

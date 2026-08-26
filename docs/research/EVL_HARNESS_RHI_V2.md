# EVL-Harness v2

本分支实现的是新版方法：**action-feedback numeric critic + recursive harness evolution**。
旧版 `rhi`、第一版数据和旧实验报告均保留不变。

## 方法

- `MatBot` 仍然是 actor，负责根据 scientific context 选择下一步 action。
- `EVL-Harness` 不替 MatBot 选择 route，而是对候选 action 输出数值：`p_success`、`uncertainty` 和 `expected_net_gain`。
- 每个 action 的 outcome 只作为事后 feedback，不进入该 action 的 runtime 输入。
- 一轮反馈先诊断当前 harness 的系统性缺陷，再提出新的 feature contract。
- 新 contract 在独立 acceptance split 上比较；只有满足风险上限且优于 incumbent 时才接受。
- 最终 test split 只在 harness 版本选择完成后评估。

## 四路数据切分

同一批第一版 `ActionRecord` 数据按固定 seed 切成：

- `train`：训练 numeric critic；
- `feedback`：生成 action-level residual 和 defect diagnosis；
- `acceptance`：验证候选 harness 是否接受；
- `test`：最终一次性报告，不能参与进化。

## Runtime 输出

`score_action_records` 的输出不含 `label` 或 `utility`：

```json
{
  "record_id": "...",
  "benchmark": "...",
  "p_success": 0.73,
  "uncertainty": 0.84,
  "expected_net_gain": 0.61,
  "cost": 0.20,
  "failure_harm": 0.15,
  "threshold": 0.70
}
```

这里的 uncertainty 是 binary success probability 的 predictive entropy；它是辅助数值信号，不是 MatBot 的 route。

## 运行

```bash
PYTHONPATH=src python3 -m harness_matsci evolve-harness \
  --data runs/paper_bootstrap_v1/normalized_actions.jsonl \
  --out runs/evl_harness_v2/report.json \
  --iterations 3 \
  --seed 7 \
  --alpha 0.10 \
  --budget-fraction 0.10
```

报告包含：

- `initial_harness` / `final_harness`；
- 每轮 `action_feedback` 的聚合诊断；
- `revisions`：候选 feature contract 与 acceptance 结果；
- `trace`：候选分数、风险和接受原因；
- `action_scores`：最终版本的 runtime numeric output；
- `test`：最终 held-out 指标。

## 与旧版的关系

- 复用旧版 `ActionRecord`、signal contract、训练器、校准器和 metrics。
- 不修改旧版 `train` / `rhi` 命令，便于严格对照。
- 新版的新增点是显式的 action feedback、defect diagnosis、候选 harness acceptance 和 oracle-free runtime score 导出。

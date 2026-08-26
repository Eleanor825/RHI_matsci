from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from harness_matsci.historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from harness_matsci.runtime_trajectory import ModelReadoutEvent, ScientificActionTrajectory, ToolCallEvent
from harness_matsci.scientific_gain import BudgetReservationValues, EmpiricalUtilityNormalizer, GainConfig, gain_targets
from harness_matsci.sd_worth_experiment import _with_task_features


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',required=True); parser.add_argument('--pilot-dir',required=True); parser.add_argument('--out',required=True); a=parser.parse_args()
    by=load_historical_tasks(a.data_dir,MAIN_MATERIAL_TASKS); sp=split_historical_tasks(by,seed=7)
    train=_with_task_features([r for t in MAIN_MATERIAL_TASKS for r in sp[t]['train']]); test=_with_task_features([r for t in MAIN_MATERIAL_TASKS for r in sp[t]['test']])
    normalizer=EmpiricalUtilityNormalizer.fit(train); cfg=GainConfig(); reservation=BudgetReservationValues.fit(train,normalizer,cfg,budget_fraction=.1)
    records={r.record_id:r for r in test}; targets={t.record_id:t for t in gain_targets(test,normalizer,cfg,reservation)}
    pilot=Path(a.pilot_dir); summaries={x['record_id']:x for x in map(json.loads,(pilot/'action_summaries.jsonl').open())}
    predictions=defaultdict(list)
    for row in map(json.loads,(pilot/'predictions.jsonl').open()): predictions[row['record_id']].append(row)
    trajectories=[]
    for record_id,summary in summaries.items():
        record=records[record_id]; rows=predictions[record_id]
        provenance={row['view_id']:row['view_provenance'] for row in rows}
        tool_calls=[]
        for view_id,tool_name in [('tool::nearest_analogs','historical_nearest_analog_retrieval'),('tool::task_statistics','historical_task_statistics')]:
            tool_calls.append(ToolCallEvent(
                event_id=f'{record_id}::{view_id}', tool_name=tool_name, tool_type='offline_database_tool',
                input_summary={'record_id':record_id,'benchmark':record.benchmark,'target_outcome_excluded':True},
                output_summary=provenance[view_id], provenance={'split':'train_only','auditable':True},
            ))
        readouts=tuple(ModelReadoutEvent(
            event_id=f"{record_id}::{row['model']}::{row['view_id']}", model=row['model'], evidence_view_id=row['view_id'],
            p_positive_gain=row['p_positive_gain'], expected_gain=row['expected_gain'], p_success=row['p_success'], assessment_confidence=row['assessment_confidence'],
        ) for row in sorted(rows,key=lambda x:(x['model'],x['view_id'])))
        target=targets[record_id]
        trajectory=ScientificActionTrajectory(
            trajectory_id=f'replay-tool-traj::{record_id}', source='closed_llm_replica_pilot_v1', online=False,
            benchmark=record.benchmark, record_id=record_id,
            visible_state={'context':record.visible_context,'evidence':record.evidence,'features':record.features},
            proposed_action={'action_type':record.action_type,'description':record.candidate_action,'cost':record.features.get('cost',0.0),'reversibility':record.features.get('reversibility',0.5)},
            tool_calls=tuple(tool_calls), model_readouts=readouts,
            harness_output={
                'action_worthiness':summary['all_view_probability'],
                'calibration_status':'uncalibrated_pilot_mean',
                'mean_predicted_gain':summary['mean_predicted_gain'],
                'internal_variance':summary['source_variance']['internal_variance'],
                'external_variance':summary['source_variance']['external_variance'],
            },
            hidden_outcome_for_evaluation_only={'hidden_from_runtime':True,'worthy':target.worthy,'gain':target.gain,'success':record.label,'utility':record.utility},
        )
        trajectories.append(trajectory.to_json())
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        for row in trajectories:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'schema_version':'scientific-action-tool-trajectory-v1','count':len(trajectories),'online':False,'source':'real API and offline retrieval calls replayed from cached artifacts','matbot_runtime':False,'route_in_harness_output':False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()

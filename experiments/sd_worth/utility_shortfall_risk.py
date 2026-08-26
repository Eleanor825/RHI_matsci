from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from harness_matsci.features import feature_names_from_records
from harness_matsci.historical import MAIN_MATERIAL_TASKS, load_historical_tasks, split_historical_tasks
from harness_matsci.metrics import binary_metrics
from harness_matsci.risk_control import select_bounded_risk_grid
from harness_matsci.scientific_gain import (
    BudgetReservationValues,
    EmpiricalUtilityNormalizer,
    GainConfig,
    ScientificShortfallScales,
    fit_gain_distribution,
    gain_targets,
)
from harness_matsci.sd_worth_experiment import (
    _calibrate_probability_temperature,
    _replace_labels,
    _temperature_scale_probabilities,
    _with_task_features,
)
from harness_matsci.training import train_gate_with_features

SEEDS=(1,7,13,21,42)
COVERAGES=(.01,.02,.05,.10,.15,.20,.30,.50)
ALPHAS=(.1,.2,.3)


def matrix(records,names):
    return np.asarray([[float(r.features.get(n,0.0)) for n in names] for r in records],dtype=float)

def platt(raw,labels,continuous=False):
    values=np.asarray(raw,dtype=float)
    if not continuous:
        values=np.log(np.clip(values,1e-6,1-1e-6)/(1-np.clip(values,1e-6,1-1e-6)))
    model=LogisticRegression(C=1e6,solver='lbfgs').fit(values.reshape(-1,1),labels)
    return model

def apply_platt(model,raw,continuous=False):
    values=np.asarray(raw,dtype=float)
    if not continuous:
        values=np.log(np.clip(values,1e-6,1-1e-6)/(1-np.clip(values,1e-6,1-1e-6)))
    return model.predict_proba(values.reshape(-1,1))[:,1]

def run(data_dir,out_dir):
    by=load_historical_tasks(data_dir,MAIN_MATERIAL_TASKS); runs=[]
    for seed in SEEDS:
        task_splits=split_historical_tasks(by,seed=seed)
        splits={k:_with_task_features([r for t in MAIN_MATERIAL_TASKS for r in task_splits[t][k]]) for k in ('train','feedback','acceptance','test')}
        cfg=GainConfig(); norm=EmpiricalUtilityNormalizer.fit(splits['train'])
        reservation=BudgetReservationValues.fit(splits['train'],norm,cfg,budget_fraction=.1)
        targets={k:gain_targets(v,norm,cfg,reservation) for k,v in splits.items()}
        shortfall=ScientificShortfallScales.fit(targets['train'])
        losses={k:[shortfall.loss(t) for t in values] for k,values in targets.items()}
        names=feature_names_from_records(splits['train'])
        scores={}

        reliability=train_gate_with_features(splits['train'],splits['feedback'],feature_names=names,epochs=90,learning_rate=.08,l2=.01,balance_benchmarks=True)
        temp=_calibrate_probability_temperature(reliability.predict_proba(splits['feedback']),[t.worthy for t in targets['feedback']])
        scores['static_reliability']={k:_temperature_scale_probabilities(reliability.predict_proba(splits[k]),temp) for k in ('acceptance','test')}

        worth_train=_replace_labels(splits['train'],targets['train']); worth_feedback=_replace_labels(splits['feedback'],targets['feedback'])
        worth=train_gate_with_features(worth_train,worth_feedback,feature_names=names,epochs=90,learning_rate=.08,l2=.01,balance_benchmarks=True)
        temp=_calibrate_probability_temperature(worth.predict_proba(worth_feedback),[t.worthy for t in targets['feedback']])
        scores['binary_budget_worthiness']={k:_temperature_scale_probabilities(worth.predict_proba(_replace_labels(splits[k],targets[k])),temp) for k in ('acceptance','test')}

        gain_model=fit_gain_distribution(splits['train'],targets['train'],splits['feedback'],targets['feedback'],feature_names=names,l2=.1)
        scores['sd_worth_gain_distribution']={k:[p.action_worthiness for p in gain_model.predict(splits[k])] for k in ('acceptance','test')}

        x={k:matrix(v,names) for k,v in splits.items()}; y={k:np.asarray([t.worthy for t in targets[k]]) for k in splits}
        classifiers={
            'extra_trees_classifier':ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=-1),
            'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,class_weight='balanced',random_state=seed),
        }
        for name,model in classifiers.items():
            model.fit(x['train'],y['train']); calibration=platt(model.predict_proba(x['feedback'])[:,1],y['feedback'])
            scores[name]={k:apply_platt(calibration,model.predict_proba(x[k])[:,1]).tolist() for k in ('acceptance','test')}
        reg=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,max_features='sqrt',random_state=seed,n_jobs=-1).fit(x['train'],np.asarray([t.gain for t in targets['train']]))
        calibration=platt(reg.predict(x['feedback']),y['feedback'],continuous=True)
        scores['extra_trees_gain_regression']={k:apply_platt(calibration,reg.predict(x[k]),continuous=True).tolist() for k in ('acceptance','test')}

        methods={}
        for method,method_scores in scores.items():
            methods[method]={}
            for alpha in ALPHAS:
                selection=select_bounded_risk_grid(method_scores['acceptance'],losses['acceptance'],coverages=COVERAGES,alpha=alpha,delta=.05)
                methods[method][str(alpha)]=evaluate(splits['test'],targets['test'],losses['test'],method_scores['test'],selection)
        runs.append({'seed':seed,'shortfall_scales':shortfall.to_json(),'methods':methods})
    summary=summarize(runs)
    report={'schema_version':'sd-worth-h3-utility-shortfall-v1','protocol':'research/evl_iclr/experiments/h3_utility_risk/protocol.md','seeds':list(SEEDS),'alphas':list(ALPHAS),'coverage_grid':list(COVERAGES),'runs':runs,'summary':summary}
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');(out/'RESULTS.md').write_text(markdown(report));return report

def evaluate(records,targets,losses,scores,selection):
    count=min(len(records),math.ceil(len(records)*selection.selected_fraction));order=sorted(range(len(scores)),key=lambda i:(-scores[i],i));selected=order[:count]
    labels=[t.worthy for t in targets]; metrics=binary_metrics(labels,scores,.5)
    total_gain=sum(targets[i].gain for i in selected)
    return {'selection':selection.to_json(),'test':{'n':len(records),'selected':count,'coverage':count/len(records),'mean_shortfall_risk':mean([losses[i] for i in selected]) if selected else 0.0,'binary_selective_risk':mean([1-labels[i] for i in selected]) if selected else 0.0,'mean_selected_gain':mean([targets[i].gain for i in selected]) if selected else 0.0,'net_gain_per_action':total_gain/len(records),'positive_total_gain':total_gain>0,'brier':metrics['brier'],'ece':metrics['ece'],'aurc':metrics['aurc']}}

def summarize(runs):
    output={}
    for method in runs[0]['methods']:
        output[method]={}
        for alpha in map(str,ALPHAS):
            output[method][alpha]={}
            for metric in ('coverage','mean_shortfall_risk','binary_selective_risk','mean_selected_gain','net_gain_per_action','brier','ece','aurc'):
                values=[r['methods'][method][alpha]['test'][metric] for r in runs]
                output[method][alpha][metric]={'mean':mean(values),'std':pstdev(values),'values':values}
            output[method][alpha]['positive_gain_seeds']=sum(r['methods'][method][alpha]['test']['positive_total_gain'] for r in runs)
            output[method][alpha]['acceptance_upper']={'mean':mean(r['methods'][method][alpha]['selection']['risk_upper_bound'] for r in runs)}
    return output

def markdown(report):
    lines=['# H3 Utility-Shortfall Selective Risk','','| Method | alpha | Coverage | Shortfall risk | Binary risk | Mean selected gain | Net gain/action | Positive seeds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,aa in report['summary'].items():
        for alpha,m in aa.items():
            lines.append(f"| {method} | {alpha} | {m['coverage']['mean']:.4f} | {m['mean_shortfall_risk']['mean']:.4f} | {m['binary_selective_risk']['mean']:.4f} | {m['mean_selected_gain']['mean']:.4f} | {m['net_gain_per_action']['mean']:.5f} | {m['positive_gain_seeds']}/5 |")
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();print(json.dumps(run(a.data_dir,a.out_dir)['summary'],indent=2))

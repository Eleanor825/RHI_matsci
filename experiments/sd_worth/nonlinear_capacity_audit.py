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
from harness_matsci.risk_control import select_risk_controlled_grid
from harness_matsci.scientific_gain import BudgetReservationValues, EmpiricalUtilityNormalizer, GainConfig, gain_targets
from harness_matsci.sd_worth_experiment import _with_task_features

SEEDS=(1,7,13,21,42)
COVERAGES=(.01,.02,.05,.10,.15,.20,.30,.50)


def matrix(records,names): return np.asarray([[r.features.get(n,0.0) for n in names] for r in records],dtype=float)
def platt(raw, labels):
    raw=np.clip(np.asarray(raw,dtype=float),1e-6,1-1e-6)
    z=np.log(raw/(1-raw)).reshape(-1,1)
    m=LogisticRegression(C=1e6,solver='lbfgs').fit(z,labels)
    return m
def apply_platt(model,raw):
    raw=np.clip(np.asarray(raw,dtype=float),1e-6,1-1e-6)
    return model.predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
def gain_platt(raw,labels):
    return LogisticRegression(C=1e6,solver='lbfgs').fit(np.asarray(raw).reshape(-1,1),labels)

def evaluate(records,targets,probs,selection):
    nsel=math.ceil(len(records)*selection.selected_fraction)
    order=np.argsort(-np.asarray(probs),kind='stable')[:nsel]
    labels=np.asarray([t.worthy for t in targets]); gains=np.asarray([t.gain for t in targets])
    bm=binary_metrics(labels.tolist(),np.asarray(probs).tolist(),.5)
    return {'coverage':nsel/len(records),'risk':float(np.mean(1-labels[order])) if nsel else 0.,'net_gain_per_action':float(np.sum(gains[order])/len(records)) if nsel else 0.,'brier':bm['brier'],'ece':bm['ece'],'aurc':bm['aurc'],'acceptance_upper':selection.risk_upper_bound}

def run(data_dir,out_dir):
 records_by=load_historical_tasks(data_dir,MAIN_MATERIAL_TASKS); runs=[]
 for seed in SEEDS:
  ts=split_historical_tasks(records_by,seed=seed)
  s={k:_with_task_features([r for t in MAIN_MATERIAL_TASKS for r in ts[t][k]]) for k in ('train','feedback','acceptance','test')}
  norm=EmpiricalUtilityNormalizer.fit(s['train']); cfg=GainConfig(); rv=BudgetReservationValues.fit(s['train'],norm,cfg,budget_fraction=.1)
  tar={k:gain_targets(v,norm,cfg,rv) for k,v in s.items()}; names=feature_names_from_records(s['train'])
  x={k:matrix(v,names) for k,v in s.items()}; y={k:np.asarray([t.worthy for t in tar[k]]) for k in s}; g=np.asarray([t.gain for t in tar['train']])
  models={
   'extra_trees_classifier':ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=-1),
   'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,class_weight='balanced',random_state=seed),
  }
  scores={}
  for name,m in models.items():
   m.fit(x['train'],y['train']); pf=m.predict_proba(x['feedback'])[:,1]; cal=platt(pf,y['feedback'])
   scores[name]={k:apply_platt(cal,m.predict_proba(x[k])[:,1]) for k in ('acceptance','test')}
  reg=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,max_features='sqrt',random_state=seed,n_jobs=-1).fit(x['train'],g)
  cal=gain_platt(reg.predict(x['feedback']),y['feedback'])
  scores['extra_trees_gain_regression']={k:cal.predict_proba(reg.predict(x[k]).reshape(-1,1))[:,1] for k in ('acceptance','test')}
  methods={}
  for name,sc in scores.items():
   methods[name]={}
   for alpha in (.1,.2,.3):
    sel=select_risk_controlled_grid(sc['acceptance'].tolist(),y['acceptance'].tolist(),coverages=COVERAGES,alpha=alpha,delta=.05)
    methods[name][str(alpha)]=evaluate(s['test'],tar['test'],sc['test'],sel)
   order=np.argsort(-sc['acceptance'],kind='stable')
   methods[name]['acceptance_grid']=[{'coverage':c,'risk':float(np.mean(1-y['acceptance'][order[:max(1,math.ceil(len(order)*c))]]))} for c in COVERAGES]
  runs.append({'seed':seed,'methods':methods})
 summary={}
 for name in runs[0]['methods']:
  summary[name]={}
  for alpha in ('0.1','0.2','0.3'):
   summary[name][alpha]={metric:{'mean':mean([r['methods'][name][alpha][metric] for r in runs]),'std':pstdev([r['methods'][name][alpha][metric] for r in runs])} for metric in ('coverage','risk','net_gain_per_action','brier','ece','aurc')}
 report={'protocol':'research/evl_iclr/experiments/h1_gain_risk/protocol_h1_3_capacity_audit.md','seeds':list(SEEDS),'runs':runs,'summary':summary}
 p=Path(out_dir); p.mkdir(parents=True,exist_ok=True); (p/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# H1.3 Nonlinear Capacity Audit','','| Method | alpha | Coverage | Risk | Net gain/action | Brier | ECE | AURC |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,aa in summary.items():
  for alpha,m in aa.items(): lines.append(f"| {name} | {alpha} | {m['coverage']['mean']:.4f} | {m['risk']['mean']:.4f} | {m['net_gain_per_action']['mean']:.5f} | {m['brier']['mean']:.4f} | {m['ece']['mean']:.4f} | {m['aurc']['mean']:.4f} |")
 (p/'RESULTS.md').write_text('\n'.join(lines)+'\n'); return report

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args(); print(json.dumps(run(a.data_dir,a.out_dir)['summary'],indent=2))

"""Fable 5.2 pilot vs Opus 5.5 confirmation: exploratory descriptive comparison (docs/fable52_pilot.md).
Run from repo root: .venv/bin/python scripts/fable52_pilot_compare.py"""
import json, numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, norm
def wil(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return f"{k}/{n}={p:.2f} [{c-h:.2f},{c+h:.2f}]"
C=json.load(open('results/phase2_confirm/confirm_summary.json'))
print('Opus HM1 mean_on', C['H_M1'].get('mean_on'), 'mean_off', C['H_M1'].get('mean_off'), 'p', C['H_M1'].get('p_one_sided'))
print('ON rate   F',wil(4,10),' O',wil(7,27),' Fisher p', round(fisher_exact([[4,6],[7,20]])[1],3))
print('ON|edited F',wil(4,4),' O',wil(7,19),' Fisher p', round(fisher_exact([[4,0],[7,12]])[1],3))
print('no-op     F',wil(6,10),' O',wil(8,27),' Fisher p', round(fisher_exact([[6,4],[8,19]])[1],3))
MF=json.load(open('results/fable52_pilot/mechanism_episodes.json')); MO=json.load(open('results/phase2_confirm/mechanism_episodes.json'))
dF=np.array([m['delta_auc'] for m in MF]); dO=np.array([m['delta_auc'] for m in MO])
print('dAUC mean/median F',dF.mean().round(3),np.median(dF).round(3),' O',dO.mean().round(3),np.median(dO).round(3),' MWU p',round(mannwhitneyu(dF,dO).pvalue,3))
print(sorted(MO[0].keys()))
for name,M in [('F',MF),('O',MO)]:
    s=[m for m in M if abs(m['low_prev_freq']-0.04)<1e-6]
    cls=lambda m: m.get('edit_class') or m.get('class')
    print(name,'0.04 stratum n',len(s),'classes',[cls(m) for m in s],' rest',[cls(m) for m in M if m not in s])

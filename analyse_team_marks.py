#!/usr/bin/env python3
"""
analyse_team_marks.py

Reproduces the analysis in:
  Yau, P. C. (2026). Do team marks reflect individual contribution? Empirical
  evidence from an undergraduate computing module. Preprint.

What it does
  * loads three cohorts of component marks (one Excel file per cohort);
  * classifies each component as solo or group-type (designed vs observed);
  * computes SD, robust SD, effective weights, dilution ratio, Spearman rho with a
    leave-one-out solo anchor, Cronbach's alpha, Ferguson's delta, Kendall tau;
  * Brown-Forsythe tests (Holm-adjusted), Pitman-Morgan tests, bootstrap CIs;
  * quartile analysis in percentile ranks and the reweighting simulation;
  * writes results.json, Figure1_rank_scatter.png and Figure2_reweighting.png.

Usage
  python analyse_team_marks.py --ay2324 AY2324.xlsx --ay2425 AY2425.xlsx \
         --ay2526 AY2526.xlsx --out results/

Input format (not included in this repository: student marks are not public)
  One .xlsx per cohort, column headers on row 3, one row per student, a sequential
  anonymous id, and the columns named in SPEC below. No names, matriculation
  numbers or team identifiers should be present.

Requirements: numpy, pandas, scipy, matplotlib, openpyxl (see requirements.txt).
Random seed is fixed (2026) so bootstrap intervals are reproducible.
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description='Reproduce the team-marks analysis.')
ap.add_argument('--ay2324',required=True);ap.add_argument('--ay2425',required=True);ap.add_argument('--ay2526',required=True)
ap.add_argument('--out',default='results');ap.add_argument('--header-row',type=int,default=3,help='1-based row holding column names (default 3)')
args=ap.parse_args();os.makedirs(args.out,exist_ok=True)
def load(path):
    d=pd.read_excel(path,header=args.header_row-1);d.columns=[str(c).strip() for c in d.columns];return d

# ---------------------------------------------------------------------------
# SPEC: published weights per component (as a fraction of the published total).
#   tuple = (weight, designed type, module-leader rating of individual attribution)
#   designed type: I = individual, G = group, M = mixed (60% group / 40% individual)
# Column names must match the headers in the Excel files.
# ---------------------------------------------------------------------------
Y={
 '2324':dict(d=load(args.ay2324),tot='total',comp={'participation':(.10,'I','not'),'quiz1':(.20,'I','strong'),'quiz2':(.25,'I','strong'),'project1':(.20,'G','fair'),'project2':(.25,'G','fair')},grade='grade'),
 '2425':dict(d=load(args.ay2425),tot='Total',comp={'participation':(.10,'I','not'),'lab':(.30,'I','fair'),'quiz1 (paper)':(.30,'I','strong'),'quiz2 (practical)':(.30,'G','fair')},grade='Grade'),
 '2526':dict(d=load(args.ay2526),tot='total',comp={'quiz1':(.20,'I','strong'),'quiz2':(.20,'I','strong'),'project':(.60,'M','fair')},grade='Grade'),
}
rng=np.random.default_rng(2026); B=5000
full={y:Y[y]['d'].copy() for y in Y}
for y,d in full.items():
    C=Y[y]['comp']; d['T']=sum(d[k]*w for k,(w,_,_) in C.items())
# OBSERVED classification = module leader's judgement (I = solo, G = group-type).
# Participation is excluded from the observed analysis.
R={'2324':dict(I=['quiz1','quiz2'],G=['project1','project2']),
   '2425':dict(I=['quiz1 (paper)'],G=['lab','quiz2 (practical)']),
   '2526':dict(I=['quiz1','quiz2'],G=['project'])}
# DESIGNED classification as published in the assessment brief (used for the designed-weighting SD).
DES={'2324':dict(I=['quiz1','quiz2'],G=['project1','project2']),
     '2425':dict(I=['lab','quiz1 (paper)'],G=['quiz2 (practical)']),
     '2526':dict(I=['quiz1','quiz2'],G=['project'])}
def prep(y,d):
    C=Y[y]['comp'];W={k:C[k][0] for k in C if k!='participation'};tot=sum(W.values())
    I,G=R[y]['I'],R[y]['G']
    out=d.copy()
    out['Tr']=sum(d[k]*W[k] for k in W)/tot
    out['Ic']=sum(d[k]*W[k] for k in I)/sum(W[k] for k in I)
    out['Gc']=sum(d[k]*W[k] for k in G)/sum(W[k] for k in G)
    return out,W,tot
def trimmed(y,d):
    cols=[k for k in Y[y]['comp'] if k!='participation']
    return d[(d[cols]>0).all(axis=1)]
res={};D={};WW={}
for y,d in full.items():
    p,W,tot=prep(y,d);D[y]=(p,trimmed(y,p));WW[y]=(W,tot)
J={'n_full':{y:len(D[y][0]) for y in D},'n_trim':{y:len(D[y][1]) for y in D}}
# reality weights
J['weights']={}
for y in D:
    W,tot=WW[y];I,G=R[y]['I'],R[y]['G']
    dI=sum(W[k] for k in DES[y]['I'])/tot
    J['weights'][y]=dict(real_I=sum(W[k] for k in I)/tot,real_G=sum(W[k] for k in G)/tot)
# component table (trimmed)
rows=[]
for y in D:
    d=D[y][1];W,tot=WW[y];T=d['Tr'];vt=T.var(ddof=1)
    solo=R[y]['I']
    for k in list(W):
        x=d[k];real='Solo' if k in solo else 'Group-type'
        anc=[s for s in solo if s!=k]
        rho=stats.spearmanr(x,d[anc].mean(axis=1))[0] if anc else np.nan
        des='Solo' if k in DES[y]['I'] else 'Group'
        if y=='2526': des='Mixed (60% group / 40% individual)'
        rows.append(dict(year=y,comp=k,des=des,real=real,w_des=Y[y]['comp'][k][0],w_real=W[k]/tot,mean=x.mean(),sd=x.std(ddof=1),uniq=int(x.nunique()),modal=x.value_counts(normalize=True).iloc[0],
            effw=np.cov(W[k]*x/tot,T)[0,1]/vt,rho=rho))
J['comp']=rows
# year summary
S={}
def q4(s): return pd.qcut(s.rank(method='first'),4,labels=False)
for y in D:
    d=D[y][1];T=d['Tr'];I=d['Ic'];G=d['Gc'];n=len(d)
    ii=[rng.integers(0,n,n) for _ in range(B)]
    sd_ci=np.percentile([T.values[i].std(ddof=1) for i in ii],[2.5,97.5])
    rho=stats.spearmanr(I,G)[0];rci=np.percentile([stats.spearmanr(I.values[i],G.values[i])[0] for i in ii[:2000]],[2.5,97.5])
    dil=T.std(ddof=1)/I.std(ddof=1);dil_ci=np.percentile([T.values[i].std(ddof=1)/I.values[i].std(ddof=1) for i in ii],[2.5,97.5])
    W,tot=WW[y];vt=T.var(ddof=1);effG=sum(np.cov(W[k]*d[k]/tot,T)[0,1] for k in R[y]['G'])/vt;nomG=sum(W[k] for k in R[y]['G'])/tot
    des_T=d['T'];K=12;g=d[Y[y]['grade']].value_counts();
    cols=[k for k in Y[y]['comp'] if d[k].std()>0]
    M=d[cols].values
    def alpha(M):k=M.shape[1];return k/(k-1)*(1-M.var(0,ddof=1).sum()/M.sum(1).var(ddof=1))
    S[y]=dict(n=n,mean=T.mean(),sd=T.std(ddof=1),sd_lo=sd_ci[0],sd_hi=sd_ci[1],mad=stats.median_abs_deviation(T,scale='normal'),iqr=np.subtract(*np.percentile(T,[75,25])),p9010=np.percentile(T,90)-np.percentile(T,10),
      solo_sd=I.std(ddof=1),dil=dil,dil_lo=dil_ci[0],dil_hi=dil_ci[1],effG=effG,nomG=nomG,ratio=effG/nomG,rho=rho,rho_lo=rci[0],rho_hi=rci[1],
      des_sd=des_T.std(ddof=1),alpha=alpha(M),alpha_ci=np.percentile([alpha(M[i]) for i in ii[:3000]],[2.5,97.5]).tolist(),ngr=int(len(g)),topshare=g.max()/n,ferg=(n**2-(g.values**2).sum())/(n**2-n**2/K),
      tau=stats.kendalltau(T,I)[0],qsame=(q4(T)==q4(I)).mean(),bottom_miss=1-((q4(T)==0)&(q4(I)==0)).sum()/(q4(I)==0).sum(),
      full_sd=D[y][0]['Tr'].std(ddof=1),full_des_sd=D[y][0]['T'].std(ddof=1))
J['year']=S
# variance tests on reality totals
T={y:D[y][1]['Tr'].values for y in D};Tf={y:D[y][0]['Tr'].values for y in D}
def vt(Tx):
    out={'omni':stats.levene(*Tx.values(),center='median')};pairs=[('2324','2526'),('2425','2526'),('2324','2425')];pr=[]
    for p,q in pairs:
        w=stats.levene(Tx[p],Tx[q],center='median');r=[Tx[q][rng.integers(0,len(Tx[q]),len(Tx[q]))].std(ddof=1)/Tx[p][rng.integers(0,len(Tx[p]),len(Tx[p]))].std(ddof=1) for _ in range(B)]
        pr.append(dict(a=q,b=p,ratio=Tx[q].std(ddof=1)/Tx[p].std(ddof=1),lo=float(np.percentile(r,2.5)),hi=float(np.percentile(r,97.5)),p=float(w.pvalue),F=float(w.statistic)))
    ps=[x['p'] for x in pr];o=np.argsort(ps);prev=0
    for k,i in enumerate(o): prev=max(prev,min(1,(3-k)*ps[i]));pr[i]['padj']=prev
    return dict(omni=dict(F=float(out['omni'].statistic),p=float(out['omni'].pvalue)),pairs=pr)
J['vt_trim']=vt(T);J['vt_full']=vt(Tf)
# Pitman-Morgan
def pm(x,y):
    r=np.corrcoef(x+y,x-y)[0,1];n=len(x);t=r*np.sqrt((n-2)/(1-r**2));return x.std(ddof=1)/y.std(ddof=1),stats.t.sf(abs(t),n-2)*2
PM=[]
for y,pairs in {'2324':[('quiz1','project1'),('quiz2','project1'),('quiz1','project2'),('quiz2','project2')],'2425':[('quiz1 (paper)','lab'),('quiz1 (paper)','quiz2 (practical)')],'2526':[('quiz1','project'),('quiz2','project')]}.items():
    for a,b in pairs:
        r,p=pm(D[y][1][a].values,D[y][1][b].values);PM.append(dict(year=y,a=a,b=b,ratio=r,p=p))
J['pm']=PM
# gap by quartile
gap=[]
for y in D:
    d=D[y][1];I=d['Ic'].values;G=d['Gc'].values;n=len(I);qq=q4(pd.Series(I)).values
    for k in range(4):
        m=qq==k;v=(G-I)[m];ci=np.percentile([ (lambda i:( (G[i]-I[i])[q4(pd.Series(I[i])).values==k]).mean())(rng.integers(0,n,n)) for _ in range(1500)],[2.5,97.5])
        gap.append(dict(year=y,q=k+1,solo=I[m].mean(),grp=G[m].mean(),gap=v.mean(),lo=ci[0],hi=ci[1]))
    sl=stats.linregress(I,G);slb=np.percentile([stats.linregress(I[i],G[i]).slope for i in [rng.integers(0,n,n) for _ in range(2000)]],[2.5,97.5])
    J.setdefault('slope',{})[y]=dict(b=sl.slope,lo=slb[0],hi=slb[1],r2=sl.rvalue**2,share_pos=float(((G-I)>0).mean()),mean_gap=float((G-I).mean()))
J['gap']=gap
rg=[]
for y in D:
    d=D[y][1];I=d['Ic'];G=d['Gc'];n=len(d)
    pi=(stats.rankdata(I)/n*100);pg=(stats.rankdata(G)/n*100);qq=q4(pd.Series(I.values)).values
    for k in range(4):
        m=qq==k
        bs=[]
        for _ in range(1500):
            i=rng.integers(0,n,n);ri=stats.rankdata(I.values[i])/n*100;rgp=stats.rankdata(G.values[i])/n*100;qi=q4(pd.Series(I.values[i])).values
            bs.append(rgp[qi==k].mean())
        rg.append(dict(year=y,q=k+1,solo_rank=pi[m].mean(),grp_rank=pg[m].mean(),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
J['rgap']=rg
# sweep
sw=[]
for y in D:
    d=D[y][1];I=d['Ic'];G=d['Gc']
    for w in np.linspace(0,1,101):
        t=(1-w)*I+w*G;sw.append(dict(year=y,w=w,sd=t.std(ddof=1),tau=stats.kendalltau(t,I)[0]))
sw=pd.DataFrame(sw);J['sweep']=[sw[(sw.year==y)&(sw.w.round(2).isin([0,.2,.3,.4,.5,.6,1.0]))].to_dict('records') for y in D]
# figures
plt.rcParams.update({'font.family':'serif','font.size':11})
sty={'2324':('-','k','AY2023/24'),'2425':('--','0.35','AY2024/25'),'2526':(':','k','AY2025/26')}
fig,ax=plt.subplots(1,2,figsize=(7.0,3.3))
for y in D:
    s_=sw[sw.year==y];ls,c,l=sty[y];ax[0].plot(s_.w*100,s_.sd,ls=ls,color=c,lw=1.8,label=l)
    sx=pd.DataFrame([dict(w=w,tau=stats.kendalltau((1-w)*D[y][1]['Ic']+w*D[y][1]['Gc'],D[y][1]['Ic'])[0]) for w in np.linspace(0,1,51)])
    ax[1].plot(sx.w*100,sx.tau,ls=ls,color=c,lw=1.8,label=l)
ax[0].set_ylabel('SD of total score');ax[0].set_ylim(0,10);ax[0].set_title('(a) Spread of total score',fontsize=10)
ax[1].set_ylabel('Kendall τ with solo composite');ax[1].set_ylim(0,1.05);ax[1].set_title('(b) Rank agreement with solo composite',fontsize=10)
for a_ in ax:
    a_.set_xlabel('Weight on group-type components (%)');a_.set_xlim(0,100);a_.spines[['top','right']].set_visible(False)
ax[0].legend(frameon=False,fontsize=9)
plt.tight_layout();plt.savefig(os.path.join(args.out,'Figure2_reweighting.png'),dpi=300)
# Figure 1: rank-rank scatter, 2x2
fig,axs=plt.subplots(2,2,figsize=(6.4,6.6));axs=axs.ravel()
for a_,y in zip(axs[:3],D):
    d=D[y][1];n=len(d);rx=stats.rankdata(d['Ic'])/n*100;ry=stats.rankdata(d['Gc'])/n*100
    a_.scatter(rx,ry,s=14,color='0.25',alpha=.55,lw=0);a_.set_title({'2324':'AY2023/24','2425':'AY2024/25','2526':'AY2025/26'}[y],fontsize=11)
    a_.text(.04,.92,f'ρ = {S[y]["rho"]:.2f}',transform=a_.transAxes,fontsize=10)
for y in D:
    d=D[y][1];n=len(d);rx=stats.rankdata(d['Ic'])/n*100;ry=stats.rankdata(d['Gc'])/n*100
    ls,c,l=sty[y];sl=stats.linregress(rx,ry);xs=np.array([0,100]);axs[3].plot(xs,sl.intercept+sl.slope*xs,ls=ls,color=c,lw=2,label=l)
axs[3].set_title('Fitted lines, all cohorts',fontsize=11);axs[3].legend(frameon=False,fontsize=9,loc='lower right')
for i,a_ in enumerate(axs):
    a_.set_xlim(0,100);a_.set_ylim(0,100);a_.set_aspect('equal');a_.spines[['top','right']].set_visible(False)
    if i>=2:a_.set_xlabel('Solo composite rank')
    if i%2==0:a_.set_ylabel('Group-type rank')
plt.tight_layout();plt.savefig(os.path.join(args.out,'Figure1_rank_scatter.png'),dpi=300)
def conv(o):
    if isinstance(o,(np.floating,)):return float(o)
    if isinstance(o,(np.integer,)):return int(o)
    return str(o)
json.dump(J,open(os.path.join(args.out,'results.json'),'w'),default=conv,indent=1)
print('Done. Outputs written to',args.out)

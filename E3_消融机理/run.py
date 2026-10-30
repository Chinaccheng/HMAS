"""E3 v11: SCFC ablations, per-role node losses and Top-10 role composition."""
from __future__ import annotations
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/uav-numba-cache')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/uav-e3-v11-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/uav-e3-v11-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,hashlib,json,math,sys,time,warnings
sys.dont_write_bytecode=True
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from UAV.experiment_config import screened_model,search_inputs,seeds
from UAV.model import NetworkState
from UAV.E2_识别准确性.run import cmlc,task_loops,batch,plotting,math_notation

OUT=ROOT/'result'
METHODS=['SCFC','no_reroute','alpha0','no_failed','degree','loop','CMLC']
LABELS=['SCFC','No rerouting','Global rerouting','No failed groups','Degree','Task loops','CMLC']
COLORS=['#C74746','#BD8D56','#9C7560','#C7AA80','#3877AD','#35775C','#9569AB']
ROLE_COLOR={'S':'#3877AD','D':'#DE8A36','I':'#428A61'}
ROLES=('S','D','I')
TOP_METHODS=('true_loss','SCFC','no_reroute','CMLC')


def tau(a,b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');v=stats.kendalltau(a,b,variant='b').statistic
    return float(v) if np.isfinite(v) else None


def mean_ci(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a):return None,None,0
    return float(a.mean()),float(stats.t.ppf(.975,len(a)-1)*stats.sem(a)) if len(a)>1 else None,len(a)


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)


def read_csv(name):
    def parse(v):
        if v=='':return None
        if v in ('True','False'):return v=='True'
        try:return int(v)
        except ValueError:
            try:return float(v)
            except ValueError:return v
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return [{k:parse(v) for k,v in row.items()} for row in csv.DictReader(f)]


def worker(index):
    model,generation=screened_model(60,index)
    draws=search_inputs(model,index,K=1000)
    ev=model.evaluate(random_inputs=draws[:,:300,:])
    assert ev.feasible_tasks.all()
    scores={'SCFC':model.scfc(ev,alpha=3,kind='node')[0],
        'no_reroute':model.scfc(ev,reroute=False,kind='node')[0],
        'alpha0':model.scfc(ev,alpha=0,kind='node')[0],
        'no_failed':model.scfc(ev,include_failed=False,kind='node')[0],
        'degree':np.asarray([model.graph().degree(i) for i in range(60)],float),
        'loop':task_loops(model)[0], 'CMLC':cmlc(model,ev)[0]}
    # Independent definition of functional weighted hyperdegree, to check H0.
    h0=np.zeros(60)
    for groups in ev.groups:
        for members,g in groups.items():h0[list(members)]+=g.efficiency_sum/(50*300)
    assert np.allclose(h0,scores['no_reroute'],atol=2e-14,rtol=0)
    scene_seed,search_seed=seeds(60,index)
    second=np.random.default_rng(search_seed+700_000_000).random((50,1000,21))[:,:,:13]
    states=[NetworkState({i},set()) for i in range(60)]
    truths=[];bases=[];afters=[]
    for d in (draws,second):
        base=model.evaluate(random_inputs=d,collect=False).F
        after=batch(model,states,d)
        for j in (0,30,59):
            scalar=model.evaluate(states[j],random_inputs=d,collect=False).F
            assert np.isclose(scalar,after[j],atol=2e-14,rtol=0)
        truths.append(base-after);bases.append(base);afters.append(after)
    # Effective hyperedges are successful member sets; failed groups do not count.
    participants=set().union(*(set(team) for teams in ev.task_combinations for team in teams))
    assert participants==set(np.flatnonzero(h0>0))
    nodes=[]
    for i,role in enumerate(model.roles):
        nodes.append({'instance':index,'node':i,'role':str(role),'participant':i in participants,
            'H0':float(h0[i]),'true_loss':float(truths[0][i]),'true_loss_2':float(truths[1][i]),
            **{m:float(scores[m][i]) for m in METHODS}})
    ablations=[{'instance':index,'method':m,'tau':tau(scores[m],truths[0]),
        'tau_truth_2':tau(scores[m],truths[1])} for m in METHODS]
    scene={'instance':index,'scene_seed':scene_seed,'search_seed':search_seed,'truth_seed_2':search_seed+700_000_000,
        'N':60,'M':50,'K':300,'truth_K':1000,'lambda':10,'Lmax':12,'alpha':3,
        'communication_edges':len(model.initial_edges),'feasible_tasks':int(ev.feasible_tasks.sum()),
        'task_positions_rejected':sum(g['rejected'] for g in generation),'F0':ev.F,'P0':ev.P,
        'truth_baseline_F':bases[0],'truth_baseline_F_2':bases[1],'noise_tau':tau(*truths),
        'SCFC_tau':ablations[0]['tau']}
    return scene,nodes,ablations


def aggregate(results):
    scenes=sorted([d[0] for d in results],key=lambda r:r['instance'])
    nodes=sorted([r for d in results for r in d[1]],key=lambda r:(r['instance'],r['node']))
    ablations=sorted([r for d in results for r in d[2]],key=lambda r:(r['instance'],METHODS.index(r['method'])))
    summary=[];tests=[]
    for m in METHODS:
        rr=[r for r in ablations if r['method']==m];mu,ci,n=mean_ci([r['tau'] for r in rr])
        summary.append({'method':m,'tau':mu,'ci95':ci,'scenes':n,
            'tau_truth_2':mean_ci([r['tau_truth_2'] for r in rr])[0]})
        if m!='SCFC':
            pairs=[(next(r['tau'] for r in ablations if r['instance']==s['instance'] and r['method']=='SCFC'),
                next(r['tau'] for r in ablations if r['instance']==s['instance'] and r['method']==m)) for s in scenes]
            diff=np.array([a-b for a,b in pairs if a is not None and b is not None])
            p=float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.
            tests.append({'comparison':'SCFC vs '+m,'n':len(diff),'mean_difference':float(diff.mean()),
                'p_two_sided':p,'SCFC_higher':bool(diff.mean()>0)})
    # Holm correction for the six paired comparisons, in addition to raw p-values.
    order=np.argsort([r['p_two_sided'] for r in tests]);running=0.
    for j,i in enumerate(order):
        running=max(running,min(1.,(len(tests)-j)*tests[i]['p_two_sided']))
        tests[i]['p_holm']=running
    role_rows=[];top_rows=[]
    for s in scenes:
        nn=[r for r in nodes if r['instance']==s['instance']]
        assert len(nn)==60 and all(np.isfinite(r[m]) for r in nn for m in METHODS+['true_loss','true_loss_2'])
        for role in ROLES:
            rr=[r for r in nn if r['role']==role and r['participant']]
            if not rr:raise ValueError(f"No participating {role} nodes in scene {s['instance']}")
            role_rows.append({'instance':s['instance'],'role':role,'total_nodes':sum(r['role']==role for r in nn),
                'participating_nodes':len(rr),**{m:float(np.mean([r[m] for r in rr]))
                    for m in ['no_reroute','SCFC','true_loss','true_loss_2']}})
        for method in TOP_METHODS:
            top=sorted(nn,key=lambda r:(-r[method],r['node']))[:10]
            assert len(top)==10
            for role in ROLES:
                count=sum(r['role']==role for r in top)
                top_rows.append({'instance':s['instance'],'method':method,'role':role,'count':count,'share':count/10})
    role_summary=[];top_summary=[]
    for role in ROLES:
        rr=[r for r in role_rows if r['role']==role]
        for method in ['no_reroute','SCFC','true_loss']:
            mu,ci,n=mean_ci([r[method] for r in rr])
            role_summary.append({'role':role,'method':method,'mean_loss':mu,'ci95':ci,'scenes':n,
                'participating_nodes':sum(r['participating_nodes'] for r in rr)})
    for method in TOP_METHODS:
        for role in ROLES:
            rr=[r['share'] for r in top_rows if r['method']==method and r['role']==role]
            mu,ci,n=mean_ci(rr)
            top_summary.append({'method':method,'role':role,'mean_share':mu,'ci95':ci,'scenes':n})
        assert np.isclose(sum(r['mean_share'] for r in top_summary if r['method']==method),1)
    for name,rows in [('scene_results.csv',scenes),('node_results.csv',nodes),('ablation_instance.csv',ablations),
        ('ablation_summary.csv',summary),('role_loss_instance.csv',role_rows),('role_loss_summary.csv',role_summary),
        ('top10_role_instance.csv',top_rows),('top10_role_summary.csv',top_summary),('wilcoxon.csv',tests)]:write_csv(name,rows)
    metadata={'design':'v11','instances':len(scenes),'N':60,'M':50,'K':300,'truth_K':1000,'truth_replicates':2,
        'lambda':10,'Lmax':12,'alpha':3,'rS':30,'rc':25,'rI':20,'feasible_task_screening':True,
        'primary_truth':'first batch; second independent batch checks noise; no averaging or switching',
        'paired_draws':'intact and all single-node failures share inputs within each truth batch',
        'role_loss_filter':'membership in union of successful effective hyperedges at construction K=300',
        'role_loss_averaging':'within each scene and role: arithmetic mean of participating nodes; then equal-weight scene mean',
        'ci95':'two-sided Student-t over independent scenes, df=n-1',
        'negative_losses':'retained; no clipping',
        'top10_selection':'all 60 nodes per scene; descending score, ascending node ID for ties; role share then mean across scenes',
        'top10_error_bars':'not displayed as design specifies percentage stacks; scene-level CI retained in source table',
        'source_model_sha256':hashlib.sha256((ROOT.parent/'model.py').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((ROOT.parent/'experiment_config.py').read_bytes()).hexdigest()}
    (OUT/'run_config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    write_csv('captions.csv',[{'figure':'图9','caption':
        f'{len(scenes)}个独立S2可行任务场景；N=60，M=50，K=300，λ=10，Lmax=12，α=3，rS=30、rc=25、rI=20。'
        '真实损失ΔF采用K′=1000重新搜索，节点失效前后共用随机数；独立计算两批真值，第一批为主结果，第二批只检查随机噪声。'
        '(a)节点Kendall τb的场景均值，误差棒为场景级95% Student-t置信区间；柱顶数字为该方法均值减完整SCFC均值。'
        '(b)各角色单节点失效的平均功能损失。仅纳入K=300构建时至少属于一条成功有效超边的节点；先在每个场景内按角色求人均值，再对场景等权平均；误差棒同(a)。'
        '浅灰为无改道H0，红色为SCFC，深灰为真实ΔF。负值保留。'
        '(c)每个场景在全部60个节点中，分别按真实ΔF、SCFC、H0及CMLC降序取前10，分值相同按节点编号升序；计算S/D/I构成后在场景间等权平均，堆积柱总和为100%。'
        '角色构成接近仅表示角色层面的一致，不等价于具体节点Top-10命中率。'}])
    print(json.dumps({'ablation':summary,'role_loss':role_summary,'top10_role':top_summary,
        'noise_tau_mean':mean_ci([r['noise_tau'] for r in scenes])[0]},ensure_ascii=False),flush=True)


def plot(qa_dir):
    summ=read_csv('ablation_summary.csv');rs=read_csv('role_loss_summary.csv');ts=read_csv('top10_role_summary.csv')
    plt=plotting()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
        'axes.titlesize':8.5,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
    # Quantitative grid: ranking ablation -> loss decomposition -> rank composition.
    fig=plt.figure(figsize=(190/25.4,94/25.4))
    axes=[fig.add_axes([x,.31,.245,.53]) for x in [.065,.391,.717]]
    a,b,c=axes
    means=[next(r['tau'] for r in summ if r['method']==m) for m in METHODS]
    cis=[next(r['ci95'] or 0 for r in summ if r['method']==m) for m in METHODS]
    a.bar(range(7),means,yerr=cis,capsize=2,color=COLORS,width=.70,error_kw={'lw':.6})
    for j in range(1,7):a.text(j,means[j]+cis[j]+.028,f'{means[j]-means[0]:+.2f}',ha='center',va='bottom',fontsize=8)
    a.set(xticks=range(7),xticklabels=LABELS,ylabel='Node Kendall τb',ylim=(-.1,1.04))
    a.tick_params(axis='x',labelrotation=90)
    for tick in a.get_xticklabels():tick.set_rotation_mode('anchor');tick.set_ha('right');tick.set_va('center')
    width=.23;series=[('no_reroute','No rerouting','#C7C7C7'),('SCFC','SCFC','#C74746'),('true_loss','True loss','#4B4B4B')]
    extents=[]
    for j,(method,label,color) in enumerate(series):
        rr=[next(r for r in rs if r['role']==role and r['method']==method) for role in ROLES]
        mu=np.array([r['mean_loss'] for r in rr]);ci=np.array([r['ci95'] or 0 for r in rr])
        b.bar(np.arange(3)+(j-1)*width,mu,width=width,color=color,label=label,
            yerr=ci,capsize=2,error_kw={'lw':.6})
        extents.extend(mu+ci);extents.extend(mu-ci)
    span=max(extents)-min(extents)
    b.axhline(0,color='#999999',lw=.6,zorder=0)
    b.set(xticks=range(3),xticklabels=ROLES,xlabel='Node role',ylabel='Mean loss per participating node',
        ylim=(min(0,min(extents))-.07*span,max(extents)+.7*span))
    b.ticklabel_format(axis='y',style='plain',useOffset=False)
    b.legend(loc='upper right',frameon=False,handlelength=1.1,labelspacing=.35)
    x=np.array([0,.85,1.70,2.55]);bottom=np.zeros(4)
    for role,color in ROLE_COLOR.items():
        yy=np.array([next(r['mean_share'] for r in ts if r['method']==m and r['role']==role) for m in TOP_METHODS])*100
        c.bar(x,yy,bottom=bottom,width=.58,color=color,label=role);bottom+=yy
    assert np.allclose(bottom,100)
    c.set(xticks=x,xticklabels=['True loss','SCFC','No rerouting','CMLC'],ylim=(0,100),xlim=(-.5,3.8),
        yticks=[0,25,50,75,100],ylabel='Role share in Top-10 (%)')
    c.tick_params(axis='x',labelrotation=90)
    for tick in c.get_xticklabels():tick.set_rotation_mode('anchor');tick.set_ha('right');tick.set_va('center')
    c.legend(loc='upper right',frameon=False,handlelength=.8,handletextpad=.4,borderaxespad=.1,labelspacing=.4)
    for x,title in zip([.065,.391,.717],['(a) Ablation accuracy','(b) Loss by node role','(c) Top-10 role composition']):
        fig.text(x,.91,title,fontsize=8.5,fontweight='bold',ha='left')
    math_notation(fig)
    skill=Path('/Users/jackpan/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(skill))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa=Path(qa_dir);qa.mkdir(parents=True,exist_ok=True)
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=['a','b','c'],
        row_groups=[{'id':'row','panels':['a','b','c']}],json_out=str(qa/'fig09.alignment.json'),strict=True)
    fig.savefig(OUT/'fig09_ablation.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/'fig09_ablation.svg',bbox_inches='tight',pad_inches=.04);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances',type=int,default=50);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--K',type=int,choices=[300],default=300)
    p.add_argument('--compute-only',action='store_true');p.add_argument('--plot-only',action='store_true')
    p.add_argument('--force',action='store_true');p.add_argument('--qa-dir',default='/private/tmp/uav-e3-v11-qa')
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.compute_only and args.plot_only:p.error('--compute-only and --plot-only cannot be combined')
    if args.plot_only:plot(args.qa_dir);return
    if args.instances<1 or args.workers<1:raise ValueError('instances/workers must be positive')
    if args.force:
        for q in OUT.iterdir():
            if q.is_file():q.unlink()
            else:raise ValueError(f'Unexpected result subfolder {q.name}')
    results=[];begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,i) for i in range(args.instances)]
        for f in as_completed(futures):
            results.append(f.result())
            print(f'E3 v11: {len(results)}/{args.instances} scenes; elapsed {time.perf_counter()-begin:.1f}s',flush=True)
    aggregate(results)
    if not args.compute_only:plot(args.qa_dir)

if __name__=='__main__':main()

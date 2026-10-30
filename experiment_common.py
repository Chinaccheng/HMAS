"""v11实验的编排、统计和绘图工具；所有物理搜索/失效/恢复仍调用根目录主模型。"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_CACHE_DIR","/private/tmp/uav-numba-cache")
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/uav-v6-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/uav-v6-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,importlib.util,inspect,json,math,sys,time,warnings
from dataclasses import replace
from collections import defaultdict
from functools import lru_cache
from . import experiment_config as scenario
import hashlib,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import networkx as nx
from scipy import stats
from .config import Config
from .model import UAVModel,NetworkState,Task,RecoveryAction,resilience

ROOT=Path(__file__).resolve().parent
FOLDERS={3:'E3_消融机理',4:'E4_定向打击',5:'E5_资源恢复',6:'E6_结构因素',7:'E7_参数敏感性'}
spec=importlib.util.spec_from_file_location('uav_e2_metrics',ROOT/'E2_识别准确性/run.py')
e2=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e2;spec.loader.exec_module(e2)
SEED=scenario.SEED
COLOR={**e2.COLOR,'greedy':'#222222','no_reroute':'#BD8D56','alpha0':'#9C7560','no_failed':'#C7AA80'}
LABEL={**e2.LABEL,'greedy':'One-step greedy','no_reroute':'No rerouting','alpha0':'Global rerouting','no_failed':'No failed groups'}
MARKER={**e2.MARKER,'greedy':None,'h':'v'}
TASKS=['Basic','S–S','D–D','Feedback','Composite']
TASK_COLORS=['#3877AD','#47A6A0','#DC9C39','#A86EAA','#CA5B58']
ATTACK_NODE=['random','degree','betweenness','loop','CMLC','SCFC']
ATTACK_EDGE=['random','betweenness','loop','u','h','CMLC','SCFC']
RECOVERY=['random','degree','betweenness','loop','CMLC','greedy','SCFC']
write_json=e2.write_json
read_json=e2.read_json
mean_ci=e2.mean_ci
order=e2.order
correlation=e2.correlation

def write_csv(out,name,rows):
    if name not in RESULT_FILES.get(Path(out).parent.name,()):return
    if not rows:return
    with (Path(out)/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

# BEGIN COMPUTE
def model_for(n,index,**changes):
    return scenario.screened_model(n,index,**changes)[0]

def draws_for(model,index,K=300,repeat=0,offset=0):
    seed=e2.seeds(model.config.node_count,index)[1]+repeat*700000000+offset
    base=np.random.default_rng(seed).random((len(model.tasks),max(1000,K),21))
    hops=model.config.max_hops+1
    if hops>21:
        extra=np.random.default_rng(seed+440000000).random((len(model.tasks),max(1000,K),hops-21))
        base=np.concatenate([base,extra],axis=2)
    return base[:,:K,:hops]

@lru_cache(maxsize=1)
def e2_sources():
    metrics=read_csv(ROOT/'E2_识别准确性/result/metrics_raw.csv')
    objects=read_csv(ROOT/'E2_识别准确性/result/object_scores_and_truth.csv')
    return metrics,objects


def cached_e2(n,index):
    metrics,objects=e2_sources();raw={}
    for kind in ['node','edge']:
        rows=[r for r in objects if r['n']==n and r['instance']==index and r['kind']==kind]
        if not rows:raise ValueError(f'Missing unified E2 truth: N={n}, scene={index}, {kind}')
        raw[kind]={'truth_losses':np.array([[r['truth_loss_1'],r['truth_loss_2']] for r in rows]),
            'noise_tau':correlation([r['truth_loss_1'] for r in rows],[r['truth_loss_2'] for r in rows])}
    return {'metrics':[r for r in metrics if r['n']==n and r['instance']==index],'raw':raw}


def construction(model,index,K=300):
    draws=draws_for(model,index,K)
    return draws,model.evaluate(random_inputs=draws)

def state_loops(model,state=None):
    state=NetworkState() if state is None else state
    n=model.config.node_count;g=model.graph(state);width=max(dict(g.degree()).values(),default=0)
    neighbors=np.full((n,width),-1,np.int32);degrees=np.zeros(n,np.int32)
    for i in g:
        row=sorted(g[i]);neighbors[i,:len(row)]=row;degrees[i]=len(row)
    ids=np.full((n,n),-1,np.int32)
    for j,(a,b) in enumerate(model.initial_edges):ids[a,b]=ids[b,a]=j
    roles=np.array([{'S':0,'D':1,'I':2}[r] for r in model.roles]);nodes=np.zeros(n,np.int64);edges=np.zeros(len(model.initial_edges),np.int64)
    for k,t in enumerate(model.tasks):
        starts=np.zeros(n,np.bool_);ends=np.zeros(n,np.bool_)
        starts[list(model.coverage_s[k]-state.failed_nodes)]=True;ends[list(model.coverage_i[k]-state.failed_nodes)]=True
        a,b,_=e2._simple_paths(neighbors,degrees,roles,starts,ends,t.requirements,t.minimum_hops+2,ids)
        nodes+=a;edges+=b
    return nodes,edges

def values(model,ev,kind,method,state=None):
    state=NetworkState() if state is None else state
    if method=='SCFC':
        a,b=model.scfc(ev,alpha=3,kind=kind)
    elif method in ['no_reroute','alpha0','no_failed']:
        kw={'no_reroute':{'reroute':False},'alpha0':{'alpha':0},'no_failed':{'include_failed':False}}[method]
        a,b=model.scfc(ev,kind=kind,**kw)
    elif method=='CMLC':
        a,b=e2.cmlc(model,ev)
        return a if kind=='node' else b
    elif method=='loop':
        a,b=state_loops(model,state)
        return a if kind=='node' else b
    elif method=='degree':
        graph=model.graph(state)
        return np.array([graph.degree(i) if i in graph else 0 for i in range(model.config.node_count)]) if kind=='node' else np.array([graph.degree(a)+graph.degree(b) if graph.has_edge(a,b) else 0 for a,b in model.initial_edges])
    elif method=='betweenness':
        graph=model.graph(state)
        if kind=='node':
            score=nx.betweenness_centrality(graph);return np.array([score.get(i,0) for i in range(model.config.node_count)])
        score=nx.edge_betweenness_centrality(graph);return np.array([score.get(e,0) for e in model.initial_edges])
    else:
        score=model.scores(ev,state)
        return getattr(score,method) if kind=='node' else np.array([getattr(score,method)[e] for e in model.initial_edges])
    return a if kind=='node' else np.array([b[e] for e in model.initial_edges])

def true_losses(model,index,kind='node',K=1000,repeats=2):
    targets=list(range(model.config.node_count)) if kind=='node' else list(model.initial_edges)
    states=[NetworkState({obj},set()) if kind=='node' else NetworkState(set(),{obj}) for obj in targets]
    result=[];baselines=[]
    for repeat in range(repeats):
        d=draws_for(model,index,K,repeat)
        before=model.evaluate(random_inputs=d,collect=False).F;baselines.append(before)
        after=e2.batch(model,states,d)
        result.append(before-after)
        for j in sorted({0,len(states)//2,len(states)-1}):
            check=model.evaluate(states[j],random_inputs=d,collect=False).F
            assert np.isclose(check,after[j],atol=2e-14,rtol=0)
    return np.array(result).T,baselines

def evaluate_row(model,state,draws,base,detail=False):
    ev=model.evaluate(state,random_inputs=draws,collect=False)
    row={'F':ev.F,'P':ev.P,'eta':ev.E,'Phi':ev.F/base.F if base.F else None,'Phi_P':ev.P/base.P if base.P else None}
    if detail:row.update(f_type=ev.f_type,f_task=ev.f_task)
    return row

def attack_path(model,index,kind,method,mode='static',repeat=0,K=300,details=False):
    draws,base=construction(model,index,K);n=model.config.node_count
    targets=list(range(n)) if kind=='node' else list(model.initial_edges);total=len(targets)
    if method=='random':ranking=np.random.default_rng(e2.seeds(n,index)[1]+900000000+repeat*997+(kind=='edge')*100000).permutation(total)
    else:ranking=order(values(model,base,kind,method))
    state=NetworkState();remaining=set(range(total));ev=base
    rows=[{'step':0,'f':0.,'F':base.F,'P':base.P,'eta':base.E,'Phi':1. if base.F else None,'Phi_P':1. if base.P else None}]
    if details:rows[0].update(f_type=base.f_type,f_task=base.f_task)
    actions=[]
    for step in range(1,math.ceil(total*.5)+1):
        if mode=='adaptive' and method!='random':
            need_collect=method in ['SCFC','CMLC','u','h']
            if step>1 and need_collect:ev=model.evaluate(state,random_inputs=draws)
            score=values(model,ev,kind,method,state)
            j=min(remaining,key=lambda j:(-score[j],j))
        else:j=int(ranking[step-1])
        remaining.remove(j);state=model.attack(state,kind,[targets[j]]);actions.append(targets[j])
        row=evaluate_row(model,state,draws,base,details);row.update(step=step,f=step/total);rows.append(row)
    return {'path':rows,'actions':actions,'objects_total':total,'baseline_F':base.F,'baseline_P':base.P}

def robustness_half(path):
    x=np.array([r['f'] for r in path]);y=np.array([r['Phi'] for r in path],float)
    assert x[0]==0 and x[-1]>=.5-1e-12 and np.all(np.diff(x)>0)
    keep=x<.5;xx=np.r_[x[keep],.5];yy=np.r_[y[keep],np.interp(.5,x,y)]
    hits=[r['f'] for r in path if r['f']<=.5 and r['Phi'] is not None and r['Phi']<.5]
    return {'R_rob':float(np.trapezoid(yy,xx)/.5) if np.isfinite(yy).all() else None,
            'f_c':hits[0] if hits else None,'f_c_capped':hits[0] if hits else .5,'f_c_censored':not bool(hits)}

def recovery_scores(model,ev):
    result={}
    for method in ['degree','betweenness','loop','CMLC','SCFC']:
        if method=='SCFC':
            a,b=model.scfc(ev,alpha=3);b=np.array([b[e] for e in model.initial_edges])
        elif method=='CMLC':a,b=e2.cmlc(model,ev)
        elif method=='loop':a,b=state_loops(model)
        else:a=values(model,ev,'node',method);b=values(model,ev,'edge',method)
        # 混合恢复时拓扑节点/链路分数单位不同，分别按原网络最大值归一化。
        if method in ['degree','betweenness']:
            a=a/max(np.max(abs(a)),1e-12);b=b/max(np.max(abs(b)),1e-12)
        result[method]=(a,b)
    return result

def damage_path(model,index,kind,attack,intensity,scores,draws,base,repeat=0):
    n=model.config.node_count;E=len(model.initial_edges);duration={.1:3,.2:6,.3:9}[intensity]
    nf=int(n*intensity*(.5 if kind=='mixed' else 1)) if kind!='edge' else 0
    ef=int(E*intensity*(.5 if kind=='mixed' else 1)) if kind!='node' else 0
    rng=np.random.default_rng(e2.seeds(n,index)[1]+1200000000+repeat*2017)
    if attack=='random':nr=rng.permutation(n);er=rng.permutation(E)
    elif attack=='area':
        centre=rng.uniform(0,model.config.region_size,2)
        nr=np.argsort(np.linalg.norm(model.positions-centre,axis=1),kind='stable')
        mid=np.array([(model.positions[a]+model.positions[b])/2 for a,b in model.initial_edges])
        er=np.argsort(np.linalg.norm(mid-centre,axis=1),kind='stable')
    else:nr=order(scores[attack][0]);er=order(scores[attack][1])
    state=NetworkState();events=[];functions={0:base.F};actions=[]
    for step in range(1,duration+1):
        ncount=int(nf*step/duration)-len(state.failed_nodes)
        if ncount:
            choices=[int(i) for i in nr if i not in state.failed_nodes][:ncount]
            state=model.attack(state,'node',choices);actions.extend(('node',i,5+step) for i in choices)
        ecount=int(ef*step/duration)-len(state.damaged_edges)
        if ecount:
            graph=model.graph(state)
            choices=[model.initial_edges[int(j)] for j in er if graph.has_edge(*model.initial_edges[int(j)])][:ecount]
            assert len(choices)==ecount
            state=model.attack(state,'edge',choices);actions.extend(('edge',e,5+step) for e in choices)
        F=model.evaluate(state,random_inputs=draws,collect=False).F
        functions[5+step]=F;events.append({'time':5+step,'F':F,'failed_nodes':sorted(state.failed_nodes),'damaged_edges':sorted(state.damaged_edges)})
    assert len(state.failed_nodes)==nf and len(state.damaged_edges)==ef
    return state,events,{'D_a':duration,'t_r':5+duration+3,'node_damage':nf,'edge_damage':ef,'actions':actions}

def recover_full(model,index,state,method,scores,draws,base,repeat=0,budget=.5):
    st=state.copy();initial=len(st.failed_nodes)+len(st.damaged_edges)
    limit=int(initial*budget+1e-9)
    if not 0<=budget<=1:raise ValueError('Recovery budget must lie in [0,1]')
    gain=draws_for(model,index,scenario.GAIN_K,offset=1500000000)
    rng=np.random.default_rng(e2.seeds(model.config.node_count,index)[1]+1600000000+repeat*4099)
    actions_log=[]
    for step in range(limit):
        actions=model.available_actions(st,len(st.failed_nodes),len(st.damaged_edges),0,2)
        if not actions:break
        calls=0
        if method=='random':choice=int(rng.integers(len(actions)))
        elif method=='greedy':
            candidates=[model.restore(st,a) for a in actions]
            fs=e2.batch(model,candidates,gain);calls=len(candidates)
            choice=max(range(len(actions)),key=lambda j:(fs[j],tuple(-v for v in actions[j].key())))
        else:
            ns,es=scores[method]
            def key(a):return (-(ns[a.target] if a.kind=='node' else es[model.edge_index[a.target]]),a.key())
            choice=min(range(len(actions)),key=lambda j:key(actions[j]))
        action=actions[choice];st=model.restore(st,action)
        F=model.evaluate(st,random_inputs=draws,collect=False).F
        actions_log.append({'step':step+1,'kind':action.kind,'target':action.target,'F':F,'candidate_calls':calls})
    assert len(actions_log)==limit
    if budget==1:assert not st.failed_nodes and not st.damaged_edges
    return actions_log

def recovery_metrics(base,events,protocol,log,budget,T=120):
    count=int((protocol['node_damage']+protocol['edge_damage'])*budget+1e-9)
    used=log[:count];tr=protocol['t_r'];func=np.full(T+1,base.F)
    for event in events:func[event['time']:]=event['F']
    for j,action in enumerate(used):
        if tr+j+1<=T:func[tr+j+1:]=action['F']
    met=resilience(np.arange(T+1),func,attack_time=5,recovery_start=tr,baseline_function=base.F,target=.9)
    assert len(used)==count and tr+count<=T
    met={k:met[k] for k in ['R_res','R_F','Phi_final']}
    met.update(actions_executed=len(used),budget_actions=count)
    return met,func/base.F if base.F else np.full_like(func,np.nan)
# END COMPUTE

def summary(rows,keys,metrics):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rs in sorted(groups.items(),key=lambda p:tuple(str(x) for x in p[0])):
        row=dict(zip(keys,key))
        for field in metrics:
            mu,ci,n=mean_ci([r.get(field) for r in rs]);row.update({field:mu,field+'_ci95':ci,field+'_n':n})
        result.append(row)
    return result

def parser(description):
    p=argparse.ArgumentParser(description)
    p.add_argument('--instances',type=int,default=50);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--random-repeats',type=int,default=20);p.add_argument('--K',type=int,default=scenario.BASE_K)
    p.add_argument('--force',action='store_true');p.add_argument('--plot-only',action='store_true');p.add_argument('--compute-only',action='store_true')
    return p

def run_jobs(number,jobs,worker,args,contract):
    out=ROOT/FOLDERS[number]/'result';out.mkdir(exist_ok=True)
    if args.K!=scenario.BASE_K:raise ValueError('Baseline K must equal experiment_config.BASE_K')
    fingerprint=hashlib.sha256(json.dumps({'contract':contract,'K':args.K,'repeats':args.random_repeats,
        'seed':SEED,'model':hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest(),
        'worker':hashlib.sha256(Path(inspect.getfile(worker)).read_bytes()).hexdigest(),
        'shared':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config':hashlib.sha256((ROOT/'experiment_config.py').read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()[:16]
    checkpoint=Path(f'/private/tmp/uav-v11-E{number}-jobs-{fingerprint}');checkpoint.mkdir(exist_ok=True)
    if args.force:
        for f in out.iterdir():
            if f.is_file():f.unlink()
            else:raise ValueError(f'Unexpected result directory: {f}')
        for f in checkpoint.glob('*.json'):f.unlink()
    result=[];pending=[]
    for j,job in enumerate(jobs):
        path=checkpoint/f'{j:05d}.json'
        if path.exists():result.append(read_json(path))
        else:pending.append((j,job))
    print(f'E{number}: {len(result)} current-run checkpoints; {len(pending)} jobs remaining',flush=True)
    begin=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,*job,args):j for j,job in pending}
        for future in as_completed(futures):
            value=future.result();write_json(checkpoint/f'{futures[future]:05d}.json',value);result.append(value)
            if len(result)%10==0 or len(result)==len(jobs):print(f'E{number}: {len(result)}/{len(jobs)}; {time.perf_counter()-begin:.1f}s',flush=True)
    write_json(out/'run_config.json',{'design':'v11','baseline':scenario.config().to_dict(),'SEED':SEED,
        'feasible_task_screening':True,'truth_K':1000,'truth_replicates':2,'greedy_search_K':scenario.GAIN_K,
        'random_repeats':args.random_repeats,'contract':contract,'fingerprint':fingerprint})
    return out,sorted(result,key=lambda d:(str(d.get('factor','')),str(d.get('value','')),str(d.get('kind','')),str(d.get('intensity','')),str(d.get('attack','')),d['instance']))


def plotting():
    plt=e2.plotting()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600})
    return plt
panel=e2.panel

def export(out,fig,name,axes=None,exemptions=()):
    e2.math_notation(fig)
    scripts=Path('/Users/jackpan/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa=Path('/private/tmp/uav-v11-qa');qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,axes=axes,exemptions=list(exemptions),
        json_out=str(qa/(name+'.alignment.json')),strict=True)
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
    plotting().close(fig)


def bands(ax,rows,x,y,methods,styles=None):
    for method in methods:
        rs=[r for r in rows if r['method']==method];xs=sorted({r[x] for r in rs});mu=[];ci=[]
        for xx in xs:
            m,c,n=mean_ci([r[y] for r in rs if r[x]==xx]);mu.append(np.nan if m is None else m);ci.append(c or 0)
        mu=np.array(mu);ci=np.array(ci);style=(styles or {}).get(method,'--' if method=='greedy' else '-')
        ax.fill_between(xs,mu-ci,mu+ci,color=COLOR.get(method,'#777777'),alpha=.10,lw=0,rasterized=True,zorder=1)
        ax.plot(xs,mu,color=COLOR.get(method,'#777777'),ls=style,lw=1.6 if method=='SCFC' else .85,
                marker=MARKER.get(method),markersize=2.5,markevery=max(1,len(xs)//10),zorder=4 if method=='SCFC' else 3)

def legend(fig,methods,ncol=4,y=.005):
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLOR[m],ls='--' if m=='greedy' else '-',marker=MARKER.get(m),
              lw=1.6 if m=='SCFC' else .85,markersize=3,label=LABEL[m]) for m in methods]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,y),ncol=ncol,frameon=False,
               handlelength=1.5,columnspacing=1.0,handletextpad=.4)



def paired_tests(rows,keys,metric,higher=True):
    output=[];groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    for key,rs in groups.items():
        methods=sorted({r['method'] for r in rs});means={m:mean_ci([r.get(metric) for r in rs if r['method']==m])[0] for m in methods}
        choices=[m for m in methods if m not in ['SCFC','random'] and means[m] is not None]
        if not choices or 'SCFC' not in methods:continue
        best=(max if higher else min)(choices,key=lambda m:means[m])
        for label,baseline in [('CMLC','CMLC'),('best',best)]:
            if baseline not in methods:continue
            a={r['instance']:r[metric] for r in rs if r['method']=='SCFC' and r.get(metric) is not None}
            b={r['instance']:r[metric] for r in rs if r['method']==baseline and r.get(metric) is not None}
            diff=np.array([a[i]-b[i] for i in sorted(a.keys()&b.keys())]);p=float(stats.wilcoxon(diff).pvalue) if len(diff) and np.any(diff) else 1.
            output.append({**dict(zip(keys,key)),'metric':metric,'comparison':label,'baseline':baseline,'n':len(diff),
                'difference':float(diff.mean()) if len(diff) else None,'p_two_sided':p,'star':bool(len(diff) and (diff.mean()>0 if higher else diff.mean()<0) and p<.05)})
    return output



def read_csv(path):
    def value(v):
        if v=='':return None
        if v in ['True','False']:return v=='True'
        try:return int(v)
        except ValueError:
            try:return float(v)
            except ValueError:return v
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return [{k:value(v) for k,v in row.items()} for row in csv.DictReader(f)]

RESULT_FILES={
 'E4_定向打击':['metrics_instance.csv','table5_robustness.csv','curves_instance.csv','efficiency_summary.csv','wilcoxon.csv','task_failure_instance.csv','task_failure_summary.csv','captions.csv'],
 'E5_资源恢复':['metrics_instance.csv','recovery_curves_instance.csv','marginal_gains_instance.csv','table6_all.csv','wilcoxon.csv','SCFC_recovery_gaps.csv','scenario_manifest.csv','captions.csv'],
 'E6_结构因素':['radius_accuracy_raw.csv','composition_instance.csv','role_instance.csv','role_summary.csv','radius_summary.csv','composition_summary.csv','wilcoxon.csv','scenario_status.csv','scenario_manifest.csv','captions.csv'],
 'E7_参数敏感性':['scale_instance.csv','scale_summary.csv','parameter_instance.csv','parameter_function_summary.csv','table7_scale.csv','scenario_status.csv','scenario_manifest.csv','captions.csv']}

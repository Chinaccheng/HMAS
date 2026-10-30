"""E2识别准确性；调用根目录model.py，保留最终表格及重绘所需数据。"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_CACHE_DIR","/private/tmp/uav-numba-cache")
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/uav-e2-v6-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/uav-e2-v6-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,json,math,sys,time,warnings
sys.dont_write_bytecode=True
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from numba import njit
from scipy import stats

ROOT=Path(__file__).resolve().parent
UAV=ROOT.parent
OUT=ROOT/'result'
sys.path.insert(0,str(UAV.parent))
from UAV.config import Config
from UAV.model import UAVModel,NetworkState,ALLOWED,RELATION_BITS

from UAV import experiment_config as scenario
SEED=scenario.SEED
NODE=['degree','betweenness','s','c','b','loop','CMLC','SCFC']
EDGE=['betweenness','u','h','loop','CMLC','SCFC']
CURVE_NODE=['random','degree','betweenness','loop','CMLC','SCFC']
CURVE_EDGE=['random','betweenness','loop','u','CMLC','SCFC']
RADAR=['degree','betweenness','loop','CMLC','SCFC']
LABEL={'degree':'Degree','betweenness':'Betweenness','loop':'Task loops','s':'s','c':'c','b':'b',
       'u':r'$u_{ij}$','h':'h','CMLC':'CMLC','SCFC':'SCFC','d_H':'Hyperdegree','random':'Random'}
COLOR={'degree':'#3877AD','betweenness':'#54A2B0','loop':'#35775C','CMLC':'#9569AB',
       'SCFC':'#C74746','random':'#8C8C8C','s':'#CB8E36','c':'#9A6C42','b':'#B69C78',
       'u':'#CB8E36','h':'#9A6C42','d_H':'#9A9A9A'}
ROLE_COLOR={'S':'#3877AD','D':'#DE8A36','I':'#428A61'}
MARKER={'degree':'o','betweenness':'s','loop':'^','CMLC':'s','SCFC':'o','random':None,'u':'D'}

def write_json(path,value):
    def clean(x):
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def write_csv(name,rows):
    if name not in ['metrics_raw.csv', 'accuracy_summary.csv', 'radar_source.csv', 'topk_summary.csv', 'table4_evaluation.csv', 'scale_stability.csv', 'wilcoxon.csv', 'object_scores_and_truth.csv', 'scenario_manifest.csv', 'captions.csv']:return
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)

def read_json(path):return json.loads(Path(path).read_text())

def mean_ci(values):
    a=np.array([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a):return None,None,0
    return float(a.mean()),float(stats.t.ppf(.975,len(a)-1)*stats.sem(a)) if len(a)>1 else None,len(a)

def config(n,K=300):
    return scenario.config(n,K)


def seeds(n,index):
    return scenario.seeds(n,index)


def order(scores,method=None,all_scores=None):
    if method=='c':return np.lexsort((np.arange(len(scores)),-np.asarray(all_scores['s']),-np.asarray(all_scores['b']),-np.asarray(scores)))
    return np.lexsort((np.arange(len(scores)),-np.asarray(scores)))

def correlation(a,b,kind='tau'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v=stats.kendalltau(a,b,variant='b').statistic if kind=='tau' else stats.spearmanr(a,b).statistic
    return float(v) if np.isfinite(v) else None

@njit(cache=False)
def _simple_paths(neighbors,degrees,roles,starts,ends,requirement,limit,edge_ids):
    """完整枚举长度有界的简单路径；可达性动态规划只作安全剪枝。"""
    n=len(roles);E=int(edge_ids.max())+1
    node=np.zeros(n,np.int64);edge=np.zeros(E,np.int64);total=0
    possible=np.zeros((limit+1,n,8),np.bool_)
    for i in range(n):
        if ends[i]:possible[0,i,0]=True
    for d in range(1,limit+1):
        possible[d]=possible[d-1]
        for i in range(n):
            if roles[i]==2:continue
            for rem in range(8):
                for q in range(degrees[i]):
                    j=neighbors[i,q];a,b=roles[i],roles[j]
                    if not ((a==0 and (b==0 or b==1)) or (a==1 and (b==0 or b==1 or b==2))):continue
                    if not (starts[j] or b==1 or (rem==0 and ends[j])):continue
                    bit=1 if a==0 and b==0 else (2 if a==1 and b==1 else (4 if a==1 and b==0 else 0))
                    if possible[d-1,j,rem & ~bit]:possible[d,i,rem]=True;break
    path=np.empty(limit+1,np.int32);remaining=np.empty(limit+1,np.int32)
    cursor=np.zeros(limit+1,np.int32);visited=np.zeros(n,np.bool_)
    for start in range(n):
        if not starts[start] or not possible[limit,start,requirement]:continue
        path[0]=start;remaining[0]=requirement;cursor[0]=0;visited[start]=True;depth=0
        while depth>=0:
            i=path[depth];rem=remaining[depth]
            if cursor[depth]>=degrees[i] or depth==limit:
                visited[i]=False;depth-=1;continue
            j=neighbors[i,cursor[depth]];cursor[depth]+=1
            if visited[j]:continue
            a,b=roles[i],roles[j]
            if not ((a==0 and (b==0 or b==1)) or (a==1 and (b==0 or b==1 or b==2))):continue
            if not (starts[j] or b==1 or (rem==0 and ends[j])):continue
            bit=1 if a==0 and b==0 else (2 if a==1 and b==1 else (4 if a==1 and b==0 else 0))
            rest=rem & ~bit
            if b==2:
                if rest==0 and ends[j]:
                    total+=1;node[j]+=1;edge[edge_ids[i,j]]+=1
                    for h in range(depth+1):node[path[h]]+=1
                    for h in range(depth):edge[edge_ids[path[h],path[h+1]]]+=1
                continue
            if depth+1>=limit or not possible[limit-depth-1,j,rest]:continue
            depth+=1;path[depth]=j;remaining[depth]=rest;cursor[depth]=0;visited[j]=True
    return node,edge,total

def task_loops(model):
    n=model.config.node_count;g=model.graph();width=max(dict(g.degree()).values())
    neighbors=np.full((n,width),-1,np.int32);degrees=np.zeros(n,np.int32)
    for i in g:
        row=sorted(g[i]);neighbors[i,:len(row)]=row;degrees[i]=len(row)
    ids=np.full((n,n),-1,np.int32)
    for j,(a,b) in enumerate(model.initial_edges):ids[a,b]=ids[b,a]=j
    roles=np.array([{'S':0,'D':1,'I':2}[r] for r in model.roles])
    nodes=np.zeros(n,np.int64);edges=np.zeros(len(model.initial_edges),np.int64);totals=[]
    for k,t in enumerate(model.tasks):
        starts=np.zeros(n,np.bool_);ends=np.zeros(n,np.bool_)
        starts[list(model.coverage_s[k])]=True;ends[list(model.coverage_i[k])]=True
        a,b,count=_simple_paths(neighbors,degrees,roles,starts,ends,t.requirements,t.minimum_hops+2,ids)
        nodes+=a;edges+=b;totals.append(count)
    return nodes,edges,totals

def cmlc(model,ev,frequency=False):
    node=np.zeros(model.config.node_count);edge=np.zeros(len(model.initial_edges));K=ev.search_count
    for k,groups in enumerate(ev.groups):
        ns=np.zeros_like(node);nc=np.zeros_like(node);es=np.zeros_like(edge);ec=np.zeros_like(edge)
        valid=[(members,g) for members,g in groups.items() if g.efficiency_sum>0]
        total=sum(g.count for _,g in valid)
        for members,g in valid:
            w=g.count/total if frequency else g.efficiency_sum/K
            ns[list(members)]+=w;nc[list(members)]+=1
            for link,(count,eta) in g.links.items():
                j=model.edge_index[link]
                es[j]+=count/total if frequency else eta/K;ec[j]+=1
        node+=np.divide(ns,nc,out=np.zeros_like(ns),where=nc>0)
        edge+=np.divide(es,ec,out=np.zeros_like(es),where=ec>0)
    return node,edge

def get_scores(model,ev):
    s=model.scores(ev);a,b=model.scfc(ev,alpha=3)
    nd={'degree':s.degree,'betweenness':s.betweenness,'s':s.s,'c':s.c,'b':s.b,'d_H':s.hyperdegree,'SCFC':a}
    ed={'betweenness':np.array([s.edge_betweenness[e] for e in model.initial_edges]),
        'u':np.array([s.u[e] for e in model.initial_edges]),'h':np.array([s.h[e] for e in model.initial_edges]),
        'SCFC':np.array([b[e] for e in model.initial_edges])}
    a,b=cmlc(model,ev);nd['CMLC']=a;ed['CMLC']=b
    nd['CMLC_q'],ed['CMLC_q']=cmlc(model,ev,True)
    a,b,counts=task_loops(model);nd['loop']=a;ed['loop']=b
    return nd,ed,counts


def batch(model,states,draws,chunk=16):
    result=[]
    for j in range(0,len(states),chunk):result.extend(model.evaluate_functions(states[j:j+chunk],random_inputs=draws))
    return np.asarray(result)

def states_for(model,kind,ranking,maxk):
    targets=list(range(model.config.node_count)) if kind=='node' else list(model.initial_edges)
    states=[]
    for k in range(1,maxk+1):
        chosen={targets[j] for j in ranking[:k]}
        states.append(NetworkState(chosen,set()) if kind=='node' else NetworkState(set(),chosen))
    return states

def worker(n,index,K,truth_K,random_repeats,curves=True):
    scene_seed,search_seed=seeds(n,index);model,generation=scenario.screened_model(n,index,K=K)
    full=scenario.search_inputs(model,index,K=max(1000,truth_K,K));draws=full[:,:K,:]
    truth_draws=[full[:,:truth_K,:],scenario.truth_inputs(model,index,1,truth_K)]
    # 预热主模型、作战环计数与批量搜索接口。
    ev=model.evaluate(random_inputs=draws);model.scfc(ev)
    task_loops(model);model.evaluate_functions([NetworkState()],random_inputs=draws)
    ev=model.evaluate(random_inputs=draws)
    nd,ed,loop_counts=get_scores(model,ev)
    result={'n':n,'instance':index,'scene_seed':scene_seed,'search_seed':search_seed,'truth_seed_2':search_seed+700000000,
            'task_positions_rejected':sum(g['rejected'] for g in generation),'manifest':scenario.manifest(model,index,ev),'config':model.config.to_dict(),'K':K,'truth_K':truth_K,
            'loop_counts_by_task':loop_counts,'roles':model.roles,
            'positions':model.positions,'tasks':[{'id':t.task_id,'position':t.position,'type':t.task_type,'requirements':t.requirements} for t in model.tasks],
            'edges':model.initial_edges,'raw':{},'metrics':[],'curves':[],'random_curves':[],
            'groups':[{'task':k,'members':sorted(m),'count':g.count,'efficiency_sum':g.efficiency_sum,
                       'links':[{'edge':e,'count':v[0],'efficiency_sum':v[1]} for e,v in g.links.items()]} for k,gs in enumerate(ev.groups) for m,g in gs.items()]}
    bases=[model.evaluate(random_inputs=d,collect=False).F for d in truth_draws]
    result['F0_truth']=bases;result['F0_construction']=ev.F
    for kind,scores,methods in [('node',nd,NODE+['d_H','CMLC_q']),('edge',ed,EDGE+['CMLC_q'])]:
        targets=list(range(n)) if kind=='node' else list(model.initial_edges)
        states=[NetworkState({a},set()) if kind=='node' else NetworkState(set(),{a}) for a in targets]
        after=[batch(model,states,d) for d in truth_draws]
        losses=np.array([bases[j]-after[j] for j in range(2)]).T
        # 固定抽查逐个主模型接口与批处理完全一致，防止加速改变真值。
        for j in sorted({0,len(states)//2,len(states)-1}):
            for repeat in range(2):
                scalar=model.evaluate(states[j],random_inputs=truth_draws[repeat],collect=False).F
                if not np.isclose(scalar,after[repeat][j],atol=2e-14,rtol=0):raise AssertionError('批处理真值与主模型逐个删除不一致')
        true=losses[:,0];noise=correlation(losses[:,0],losses[:,1]);true_order=order(true)
        raw={'scores':scores,'truth_losses':losses,'after_F':np.array(after).T,
             'noise_tau':noise,'objects':targets}
        result['raw'][kind]=raw
        for method in methods:
            values=np.array(scores[method]);ranking=order(values,method,scores)
            phi=None
            if curves:
                st=states_for(model,kind,ranking,5)[-1]
                phi=model.evaluate(st,random_inputs=truth_draws[0],collect=False).F/bases[0] if bases[0] else None
            result['metrics'].append({'n':n,'instance':index,'kind':kind,'method':method,
                'tau':correlation(values,true),'rho':correlation(values,true,'rho'),
                'hit10':len(set(ranking[:10])&set(true_order[:10]))/10,
                'false5':float(np.mean(true[ranking[:5]]<.01*bases[0])), 'Phi5':phi,
                'mae':float(np.mean(abs(values-true))) if method=='SCFC' else None,
                'noise_tau':noise,
                'F0':bases[0]})
        if curves:
            curve_methods=CURVE_NODE if kind=='node' else CURVE_EDGE;maxk=10 if kind=='node' else 20
            curve_states=[];keys=[]
            for method in curve_methods:
                repeats=range(random_repeats) if method=='random' else range(1)
                for repeat in repeats:
                    ranking=np.random.default_rng(search_seed+800000000+(0 if kind=='node' else 100000)+repeat).permutation(len(targets)) if method=='random' else order(scores[method])
                    curve_states.extend(states_for(model,kind,ranking,maxk))
                    keys.extend((method,repeat,k) for k in range(1,maxk+1))
            vals=batch(model,curve_states,truth_draws[0])
            for (method,repeat,k),F in zip(keys,vals):
                result['random_curves' if method=='random' else 'curves'].append({'n':n,'instance':index,'kind':kind,'method':method,
                                'repeat':repeat,'k':k,'F':F,'Phi':F/bases[0] if bases[0] else None})
            for k in range(1,maxk+1):
                rows=[r for r in result['random_curves'] if r['kind']==kind and r['k']==k]
                result['curves'].append({'n':n,'instance':index,'kind':kind,'method':'random','k':k,
                                        'F':float(np.mean([r['F'] for r in rows])),
                                        'Phi':float(np.mean([r['Phi'] for r in rows])) if bases[0] else None})
    return result



def compute(args):
    OUT.mkdir(exist_ok=True)
    jobs=[(60,i,args.K,args.truth_K,args.random_repeats,True) for i in range(args.instances)]
    jobs += [(n,i,args.K,500,args.random_repeats,False) for n in [30,120] for i in range(args.scale_instances)]
    data=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,*job) for job in jobs]
        for i,f in enumerate(as_completed(futures),1):
            data.append(f.result())
            if i%10==0 or i==len(jobs):print(f'E2: {i}/{len(jobs)}',flush=True)
    return sorted(data,key=lambda d:(d['n'],d['instance']))

def aggregate(data):
    rows=[r for d in data for r in d['metrics']]
    curves=[r for d in data for r in d['curves']]
    write_csv('metrics_raw.csv',rows)
    write_csv('topk_raw.csv',curves)
    write_csv('random_topk_repeats.csv',[r for d in data for r in d['random_curves']])
    objects=[]
    for d in data:
        for kind,raw in d['raw'].items():
            for j,obj in enumerate(raw['objects']):
                objects.append({'n':d['n'],'instance':d['instance'],'kind':kind,'object':json.dumps(obj),
                    'role':d['roles'][j] if kind=='node' else '',
                    'truth_loss_1':raw['truth_losses'][j][0],'truth_loss_2':raw['truth_losses'][j][1],
                    **{m:v[j] for m,v in raw['scores'].items()}})
    write_csv('object_scores_and_truth.csv',objects)
    dimensions=['tau','rho','hit10','false5','Phi5','mae','noise_tau']
    summary=[]
    for n,kind,method in sorted({(r['n'],r['kind'],r['method']) for r in rows}):
        selected=[r for r in rows if (r['n'],r['kind'],r['method'])==(n,kind,method)]
        result={'n':n,'kind':kind,'method':method}
        for metric in dimensions:
            mu,ci,count=mean_ci([r[metric] for r in selected])
            result.update({metric:mu,metric+'_ci95':ci,metric+'_n':count})
        summary.append(result)
    write_csv('accuracy_summary.csv',summary)
    stability=scale_stability(rows)
    write_csv('scale_stability.csv',stability)
    curve_summary=[]
    for kind,method,k in sorted({(r['kind'],r['method'],r['k']) for r in curves}):
        mu,ci,count=mean_ci([r['Phi'] for r in curves if (r['kind'],r['method'],r['k'])==(kind,method,k)])
        curve_summary.append({'kind':kind,'method':method,'k':k,'Phi':mu,'ci95':ci,'instances':count})
    write_csv('topk_summary.csv',curve_summary)
    tests=[]
    for kind,methods in [('node',NODE),('edge',EDGE)]:
        for metric,direction in [('tau',1),('hit10',1),('false5',-1),('Phi5',-1)]:
            def avg(method):return next(r[metric] for r in summary if r['n']==60 and r['kind']==kind and r['method']==method)
            best=max([m for m in methods if m!='SCFC'],key=lambda m:direction*avg(m))
            for comparison,baseline in [('CMLC','CMLC'),('best_baseline',best)]:
                a={r['instance']:r[metric] for r in rows if r['n']==60 and r['kind']==kind and r['method']=='SCFC'}
                b={r['instance']:r[metric] for r in rows if r['n']==60 and r['kind']==kind and r['method']==baseline}
                pairs=[(a[i],b[i]) for i in sorted(a.keys()&b.keys()) if a[i] is not None and b[i] is not None]
                diff=np.array([x-y for x,y in pairs])
                pvalue=float(stats.wilcoxon(diff,zero_method='wilcox',alternative='two-sided').pvalue) if np.any(diff) else 1.
                tests.append({'kind':kind,'metric':metric,'comparison':comparison,'baseline':baseline,'n_pairs':len(pairs),
                              'mean_difference_SCFC_minus_baseline':float(diff.mean()),'p_two_sided':pvalue,
                              'SCFC_better':bool(direction*diff.mean()>0),'star':bool(direction*diff.mean()>0 and pvalue<.05)})
    for kind in ['node','edge']:
        subset=[r for r in tests if r['kind']==kind];ranked=sorted(subset,key=lambda r:r['p_two_sided']);last=0.
        for j,r in enumerate(ranked):last=max(last,min(1.,(len(ranked)-j)*r['p_two_sided']));r['p_holm']=last
    write_csv('wilcoxon.csv',tests)
    table=[]
    for kind,methods in [('node',NODE),('edge',EDGE)]:
        for metric in ['tau','hit10','false5','Phi5','stability']:
            for method in methods:
                if metric=='stability':
                    r=next(r for r in stability if r['kind']==kind and r['method']==method)
                    mu=r['stability'];ci=None;low=r['ci95_low'];high=r['ci95_high']
                else:
                    r=next(r for r in summary if r['n']==60 and r['kind']==kind and r['method']==method)
                    mu=r[metric];ci=r[metric+'_ci95'];low=mu-ci if ci is not None else None;high=mu+ci if ci is not None else None
                star=method=='SCFC' and any(r['kind']==kind and r['metric']==metric and r['comparison']=='best_baseline' and r['star'] for r in tests)
                table.append({'kind':kind,'dimension':metric,'method':method,'value':mu,'ci95':ci,'ci95_low':low,'ci95_high':high,'star':star})
    write_csv('table4_evaluation.csv',table)
    radar=radar_values(summary,stability)
    write_csv('radar_source.csv',radar)
    return summary,curve_summary,stability,tests,radar,table

def scale_stability(rows):
    """Range-based agreement of scene-mean Kendall tau across the three scales."""
    stability=[];rng=np.random.default_rng(SEED+660)
    for kind,methods in [('node',NODE+['d_H','CMLC_q']),('edge',EDGE+['CMLC_q'])]:
        for method in methods:
            by_scale=[np.array([r['tau'] for r in rows if r['n']==n and r['kind']==kind and r['method']==method and r['tau'] is not None]) for n in [30,60,120]]
            means=np.array([v.mean() for v in by_scale])
            value=1-float(np.ptp(means))
            boots=[]
            for _ in range(2000):
                mus=np.array([rng.choice(v,len(v),replace=True).mean() for v in by_scale])
                boots.append(1-float(np.ptp(mus)))
            stability.append({'kind':kind,'method':method,'tau_S1':means[0],'tau_S2':means[1],'tau_S3':means[2],
                              'stability':value,'ci95_low':float(np.quantile(boots,.025)) if boots else None,
                              'ci95_high':float(np.quantile(boots,.975)) if boots else None,
                              'definition':'1 - (max(tau_S1, tau_S2, tau_S3) - min(tau_S1, tau_S2, tau_S3)); tau is the scene mean within each scale',
                              'bootstrap':'2000 independent instance resamples within each scale'})
    return stability

def radar_values(summary,stability):
    """Direct metric values; only negative ranking accuracy is floored as requested."""
    radar=[]
    for method in RADAR:
        r=next(r for r in summary if r['n']==60 and r['kind']=='node' and r['method']==method)
        s=next(r for r in stability if r['kind']=='node' and r['method']==method)
        for dim,value,definition in [
            ('Rank agreement',max(r['tau'],0),'max(mean scene Kendall tau at S2, 0)'),
            ('Top-10 hit',r['hit10'],'mean scene Top-10 hit rate at S2'),
            ('Low false positives',1-r['false5'],'1 - mean scene false-positive rate at S2'),
            ('Task impact',1-r['Phi5'],'1 - mean Phi after jointly removing Top-5 nodes at S2'),
            ('Scale stability',s['stability'],s['definition'])]:
            radar.append({'method':method,'dimension':dim,'radar_value':value,'definition':definition,
                'normalization':'none; direct metric value'})
    return radar


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    for suffix in ['', ' Bold', ' Italic', ' Bold Italic']:
        p=Path('/System/Library/Fonts/Supplemental')/f'Times New Roman{suffix}.ttf'
        if p.exists():font_manager.fontManager.addfont(str(p))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    matplotlib.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'mathtext.fontset':'stix'})
    import matplotlib.pyplot as plt
    plt.rcParams['mathtext.fontset'] = 'stix'
    return plt

def panel(ax,label,title):
    ax.annotate(f'({label})',xy=(0,1),xycoords='axes fraction',xytext=(0,8),textcoords='offset points',
                ha='left',va='bottom',weight='bold',fontsize=9)
    ax.annotate(title,xy=(0,1),xycoords='axes fraction',xytext=(16,8),textcoords='offset points',
                ha='left',va='bottom',weight='bold',fontsize=9)
    ax.tick_params(length=2,pad=2)

def math_notation(fig):
    """Format displayed scientific symbols without changing data or existing math spans."""
    import re,matplotlib
    from matplotlib.text import Text
    matplotlib.rcParams['mathtext.fontset']='stix'
    tokens={'τb':r'$\tau_b$','τ_b':r'$\tau_b$','ΦP':r'$\Phi_P$','Φ_P':r'$\Phi_P$',
        'RF':r'$R_F$','R_F':r'$R_F$','Rrob':r'$R_{\mathrm{rob}}$','R_rob':r'$R_{\mathrm{rob}}$',
        'pS':r'$p_S$','pD':r'$p_D$','p_S':r'$p_S$','p_D':r'$p_D$',
        '1/J':r'$1/J_k$','J_k':r'$J_k$','Δτ':r'$\Delta\tau_b$',
        'τ':r'$\tau$','Φ':r'$\Phi$','η':r'$\bar{\eta}$','α':r'$\alpha$','λ':r'$\lambda$',
        'F(0)':r'$F(0)$','P(0)':r'$P(0)$','|e|':r'$|e|$'}
    words={'J':r'$J_k$','K':r'$K$','N':r'$N$','rc':r'$r_c$','lambda':r'$\lambda$'}
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(tokens,key=len,reverse=True))+r'|\b(?:J|K|N|rc|lambda)\b')
    for artist in fig.findobj(match=Text):
        artist.set_math_fontfamily('stix')
        parts=re.split(r'(\$[^$]*\$)',artist.get_text())
        for j in range(0,len(parts),2):
            parts[j]=pattern.sub(lambda m:tokens[m.group()] if m.group() in tokens else words[m.group()],parts[j])
        artist.set_text(''.join(parts))

def export(fig,name,axes=None,exemptions=()):
    math_notation(fig)
    sys.path.insert(0,'/Users/jackpan/.codex/skills/nature-figure/scripts')
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa=Path('/private/tmp/uav-v11-qa');qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,axes=axes,exemptions=list(exemptions),
        json_out=str(qa/(name+'.alignment.json')),strict=True)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04);plotting().close(fig)


def link_type(a,b,roles):
    pair=''.join(sorted([roles[a],roles[b]]))
    return {'SS':'S–S','DS':'S–D / D–S','DD':'D–D','DI':'D–I'}.get(pair,'Unused')

def render_prediction(data,summary):
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    ds=[d for d in data if d['n']==60];typical=next(d for d in ds if d['instance']==0)
    fig,axs=plt.subplots(1,3,figsize=(190/25.4,92/25.4))
    fig.subplots_adjust(left=.075,right=.93,bottom=.29,top=.84,wspace=.62)
    edge_colors={'S–S':'#3877AD','S–D / D–S':'#54A2B0','D–D':'#A4794B','D–I':'#428A61','Unused':'#AEAEAE'}
    for ax,kind,title in zip(axs[:2],['node','edge'],['Nodes: SCFC','Links: SCFC']):
        allx=[];ally=[]
        for typical_flag in [False,True]:
            selected=[d for d in ds if (d['instance']==0)==typical_flag]
            x=np.concatenate([np.array(d['raw'][kind]['truth_losses'])[:,0] for d in selected])
            y=np.concatenate([d['raw'][kind]['scores']['SCFC'] for d in selected])
            types=np.concatenate([d['roles'] if kind=='node' else [link_type(a,b,d['roles']) for a,b in d['edges']] for d in selected])
            for key,color in (ROLE_COLOR if kind=='node' else edge_colors).items():
                mask=types==key
                ax.scatter(x[mask],y[mask],s=9 if typical_flag else 3.5,color=color,
                           alpha=1 if typical_flag else .12,edgecolors='none',rasterized=True,zorder=3 if typical_flag else 2)
            allx.extend(x);ally.extend(y)
        lo=min(min(allx),min(ally));hi=max(max(allx),max(ally));span=hi-lo
        ax.plot([lo,hi],[lo,hi],'--',color='#333333',lw=.8,zorder=1)
        ax.set(xlim=(lo-.06*span,hi+.06*span),ylim=(lo-.06*span,hi+.06*span),
               xlabel='True functional loss',ylabel='SCFC predicted loss')
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
        r=next(r for r in summary if r['n']==60 and r['kind']==kind and r['method']=='SCFC')
        ax.text(.96,.045,f'τ = {r["tau"]:.3f}\nMAE = {r["mae"]:.4f}',
                transform=ax.transAxes,ha='right',va='bottom',fontsize=8,linespacing=1.2)
        panel(ax,chr(97+list(axs).index(ax)),title)
    ax=axs[2];allx=[];ally=[]
    for is_typical in [False,True]:
        selected=[d for d in ds if (d['instance']==0)==is_typical]
        x=np.concatenate([np.array(d['raw']['node']['truth_losses'])[:,0] for d in selected])
        y=np.concatenate([d['raw']['node']['scores']['CMLC'] for d in selected])
        roles=np.concatenate([d['roles'] for d in selected]);allx.extend(x);ally.extend(y)
        for role,color in ROLE_COLOR.items():
            mask=(roles==role)&(y>0)
            ax.scatter(x[mask],y[mask],s=9 if is_typical else 3.5,color=color,
                alpha=1 if is_typical else .12,edgecolors='none',rasterized=True,zorder=3 if is_typical else 2)
    xspan=np.ptp(allx);positive=np.asarray(ally)[np.asarray(ally)>0]
    ax.set_yscale('log')
    ax.set(xlim=(min(allx)-.06*xspan,max(allx)+.06*xspan),
        ylim=(positive.min()/1.35,positive.max()*3.0),
        xlabel='True functional loss',ylabel=r'$\mathrm{CMLC}_i$')
    rc=next(r for r in summary if r['n']==60 and r['kind']=='node' and r['method']=='CMLC')
    ax.text(.96,.96,f'τ = {rc["tau"]:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    panel(ax,'c','Nodes: CMLC')
    role_handles=[Line2D([],[],ls='',marker='o',color=c,label=r,markersize=4) for r,c in ROLE_COLOR.items()]
    edge_handles=[Line2D([],[],ls='',marker='o',color=c,label=r,markersize=4) for r,c in edge_colors.items()]
    fig.legend(handles=role_handles,loc='lower center',bbox_to_anchor=(.185,.07),ncol=3,frameon=False,handletextpad=.3,columnspacing=.7)
    fig.legend(handles=edge_handles,loc='lower center',bbox_to_anchor=(.50,.005),ncol=3,frameon=False,handletextpad=.3,columnspacing=.7)
    fig.legend(handles=role_handles,loc='lower center',bbox_to_anchor=(.825,.07),ncol=3,frameon=False,handletextpad=.3,columnspacing=.7)
    export(fig,'fig06_prediction',axes=list(axs))

def render_accuracy(data,summary,radar):
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig,axs=plt.subplots(1,3,figsize=(190/25.4,92/25.4))
    fig.subplots_adjust(left=.075,right=.96,bottom=.34,top=.81,wspace=.55)
    rows=[r for d in data if d['n']==60 for r in d['metrics']]
    plot_methods=[['degree','betweenness','loop','s','c','b','CMLC','SCFC'],
                  ['betweenness','loop','u','h','CMLC','SCFC']]
    for ax,kind,methods,title in zip(axs[:2],['node','edge'],plot_methods,['Node rank agreement','Link rank agreement']):
        values=[[r['tau'] for r in rows if r['kind']==kind and r['method']==m and r['tau'] is not None] for m in methods]
        bp=ax.boxplot(values,patch_artist=True,widths=.62,showfliers=True,
                      medianprops={'color':'#333333','linewidth':.9},whiskerprops={'linewidth':.6},capprops={'linewidth':.6},
                      flierprops={'marker':'.','markersize':2,'markeredgecolor':'#555555'})
        for patch,m in zip(bp['boxes'],methods):patch.set_facecolor(COLOR[m]);patch.set_alpha(.65);patch.set_linewidth(.6)
        noise=next(r['noise_tau'] for r in summary if r['n']==60 and r['kind']==kind and r['method']=='SCFC')
        ax.axhline(noise,color='#888888',ls='--',lw=.8)
        ax.set(xticks=range(1,len(methods)+1),xticklabels=[({'s':r'$s_i$','c':r'$c_i$','b':r'$b_i$','u':r'$u_{ij}$','h':r'$h_{ij}$'}.get(m,LABEL[m]) if not (kind=='edge' and m=='CMLC') else 'CMLC (ext.)') for m in methods],ylabel='Kendall τb',ylim=(-.32,1.03))
        ax.tick_params(axis='x',labelrotation=90)
        panel(ax,chr(97+list(axs).index(ax)),title)
    grid=axs[2].get_subplotspec();cell=axs[2].get_position();axs[2].remove();polar=fig.add_subplot(grid,projection='polar')
    polar.set_position([cell.x0+cell.width*.14,cell.y0+cell.height*.14,cell.width*.72,cell.height*.72])
    dimensions=list(dict.fromkeys(r['dimension'] for r in radar));angles=np.linspace(0,2*np.pi,5,endpoint=False)
    polar.set_theta_offset(np.pi/2);polar.set_theta_direction(-1)
    for method in RADAR:
        values=[next(r['radar_value'] for r in radar if r['method']==method and r['dimension']==dim) for dim in dimensions]
        polar.plot(np.r_[angles,angles[0]],np.r_[values,values[0]],color=COLOR[method],lw=1.4 if method=='SCFC' else .8,
                   marker=MARKER[method],markersize=2.5,label=LABEL[method],rasterized=False)
        if method=='SCFC':polar.fill(angles,values,color=COLOR[method],alpha=.08,rasterized=True)
    polar.set(ylim=(0,1),yticks=[.5,1],yticklabels=[],xticks=angles,
              xticklabels=['Ranking','Top-10\nhit','Low false\npositives','Task\nimpact','Scale\nstability'])
    # Manual radial labels avoid Matplotlib's automatic alignment crossing the rings.
    for radius in [.5,1]:
        polar.annotate(f'{radius:g}',xy=(np.deg2rad(10),radius),xytext=(0,-5 if radius==.5 else -1),
            textcoords='offset points',ha='left' if radius==.5 else 'right',va='top',fontsize=7,zorder=6)
    polar.tick_params(axis='x',pad=5)
    for label,ha,va in zip(polar.get_xticklabels(),['center','left','left','right','right'],['top','center','top','top','center']):
        label.set_ha(ha);label.set_va(va)
    polar.grid(color='#D8D8D8',lw=.5);polar.spines['polar'].set_linewidth(.6)
    # 极坐标专门用于多维概览，不与两个笛卡尔分布面板比较绘图区宽高。
    fig.text(cell.x0,cell.y1+8/(92/25.4*72),'(c)  Node overview',ha='left',va='bottom',weight='bold',fontsize=9)
    handles=[Line2D([],[],color=COLOR[m],marker=MARKER[m],lw=1.3 if m=='SCFC' else .8,markersize=3,label=LABEL[m]) for m in RADAR]
    handles.append(Line2D([],[],color='#888888',ls='--',lw=.8,label='Truth repeatability'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.005),ncol=3,frameon=False,columnspacing=1.3,handlelength=1.5)
    export(fig,'fig07_accuracy',axes=list(axs[:2]),exemptions=[])

def render_topk(curves):
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,77/25.4))
    fig.subplots_adjust(left=.08,right=.97,bottom=.28,top=.83,wspace=.25)
    allhigh=max(r['Phi']+(r['ci95'] or 0) for r in curves);alllow=min(r['Phi']-(r['ci95'] or 0) for r in curves)
    ylim=(max(0,alllow-.04),max(1.02,allhigh+.03))
    for ax,kind,methods,title in zip(axs,['node','edge'],[CURVE_NODE,CURVE_EDGE],['Node removal','Link removal']):
        for method in methods:
            rows=sorted([r for r in curves if r['kind']==kind and r['method']==method],key=lambda r:r['k'])
            x=np.array([r['k'] for r in rows]);y=np.array([r['Phi'] for r in rows]);ci=np.array([r['ci95'] or 0 for r in rows])
            ax.fill_between(x,y-ci,y+ci,color=COLOR[method],alpha=.10,rasterized=True,lw=0,zorder=1)
            ax.plot(x,y,color=COLOR[method],lw=1.6 if method=='SCFC' else .85,
                    ls='--' if kind=='edge' and method=='CMLC' else '-',marker=MARKER.get(method),
                    markersize=3 if method=='SCFC' else 2.2,markevery=1 if kind=='node' else 2,zorder=4 if method=='SCFC' else 3)
        ax.set(xlim=(1,10 if kind=='node' else 20),ylim=ylim,xlabel='Jointly removed nodes' if kind=='node' else 'Jointly removed links',ylabel='Functional retention Φ')
        ax.set_xticks([1,3,5,7,10] if kind=='node' else [1,5,10,15,20])
        panel(ax,chr(97+list(axs).index(ax)),title)
    methods=list(dict.fromkeys(CURVE_NODE+CURVE_EDGE))
    handles=[Line2D([],[],color=COLOR[m],marker=MARKER.get(m),lw=1.6 if m=='SCFC' else .85,
                    markersize=3,label='CMLC / CMLC extension' if m=='CMLC' else LABEL[m]) for m in methods]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.005),ncol=4,frameon=False,columnspacing=1.2,handlelength=1.5)
    export(fig,'fig08_topk')

def render(data,args):
    summary,curves,stability,tests,radar,table=aggregate(data)
    write_csv('scenario_manifest.csv',[{'n':d['n'],'instance':d['instance'],'scene_seed':d['scene_seed'],
        'search_seed':d['search_seed'],'K':d['K'],'truth_K':d['truth_K'],
        'task_positions_rejected':d['task_positions_rejected'],
        'F0':d['F0_construction'],'truth_baseline_F':d['F0_truth'][0],**d['manifest'],
        **{k:d['config'][k] for k in ['priority_weight','max_hops','sensing_radius','communication_radius','execution_radius']}} for d in data])
    write_json(OUT/'run_config.json',{'design':'v11','instances':len([d for d in data if d['n']==60]),
        'baseline':scenario.config().to_dict(),'SEED':SEED,'feasible_task_screening':True,
        'truth':'two independent batches; first primary; intact and failures share draws; fixed 21-column prefixes',
        'scale_cohort':'same seeds and tasks as E7; S1/S3 truth_K500','timing_experiment':False})
    packed=[]
    for d in data:
        if d['n']!=60:continue
        v={k:d[k] for k in ['n','instance','roles','edges','F0_truth']};v['raw']={}
        for kind in ['node','edge']:
            raw=d['raw'][kind]
            v['raw'][kind]={'truth_losses':[[x[0]] for x in raw['truth_losses']],
                'scores':{k:raw['scores'][k] for k in (['SCFC','CMLC'] if kind=='node' else ['SCFC'])}}
        packed.append(v)
    write_json(OUT/'prediction_plot.json',packed)
    if not args.compute_only:plot()

def plot():
    data=read_json(OUT/'prediction_plot.json')
    metrics=read_csv(OUT/'metrics_raw.csv')
    for d in data:d['metrics']=[r for r in metrics if r['n']==60 and r['instance']==d['instance']]
    s=read_csv(OUT/'accuracy_summary.csv')
    stability=scale_stability(metrics);write_csv('scale_stability.csv',stability)
    table=read_csv(OUT/'table4_evaluation.csv')
    for r in table:
        if r['dimension']=='stability':
            v=next(v for v in stability if v['kind']==r['kind'] and v['method']==r['method'])
            r.update(value=v['stability'],ci95=None,ci95_low=v['ci95_low'],ci95_high=v['ci95_high'])
    write_csv('table4_evaluation.csv',table)
    radar=radar_values(s,stability);write_csv('radar_source.csv',radar)
    render_prediction(data,s);render_accuracy(data,s,radar);render_topk(read_csv(OUT/'topk_summary.csv'))
    write_captions()

def write_captions():
    nodes=[r for r in read_csv(OUT/'object_scores_and_truth.csv') if r['n']==60 and r['kind']=='node']
    omitted=[r for r in nodes if r['CMLC']==0]
    assert all(r['d_H']==0 for r in omitted)
    zero_truth=sum(abs(r['truth_loss_1'])<1e-12 for r in omitted)
    common='50个独立S2可行任务场景，N=60、M=50、K=300、λ=10、Lmax=12、α=3；真值采用K′=1000重新搜索。固定第一批为主真值，第二批独立真值仅检查重复一致性；每批失效前后共用随机输入。'
    write_csv('captions.csv',[
        {'figure':'图6','caption':common+'场景0的节点与链路为实心点，其余场景半透明叠加，(a)(b)展示全部点。SCFC面板虚线为y=x；(a)(b)仅标注各场景τ与MAE的均值，两个坐标轴采用两轴数据并集的相同范围；(c)按S/D/I角色着色，纵轴采用对数坐标；省略CMLC=0的'+str(len(omitted))+'个节点（共'+str(len(nodes))+'个），这些节点在K=300构建的超边中参与度为0。其主真值损失在K′=1000评估中有'+str(zero_truth)+'个恰为0，其余并非严格为0，不能将两种搜索预算下的零值等同。右上角τ仍为各场景全部节点（含省略节点）的排序相关均值。'},
        {'figure':'图7','caption':common+'箱体为场景τb的第25至75百分位，中线为中位数，须线范围为1.5 IQR，超出者单独显示；灰色虚线为两批真值的τb场景均值。图9(a)与本图使用相同的逐场景τb，图9柱高为均值。链路CMLC为扩展版本，标记为CMLC (ext.)。雷达图五个维度直接采用：排序准确性max(τ,0)、Top-10命中率原值、1−误判率、1−Φ（联合移除Top-5节点后）、多规模稳定性1−(τ_max−τ_min)。τ_max和τ_min取E7相同S1/S2/S3样本在各规模内的场景平均τ之最大值与最小值；不再进行min-max或(τ+1)/2归一化。0.5与1为径向刻度环；只有SCFC使用红色填充，其余策略仅画线。表4中的稳定性及其2000次分规模场景bootstrap置信区间同步采用这一新定义。'},
        {'figure':'图8','caption':common+'每个场景内各策略使用相同的真值搜索输入。随机排序在每场景重复20次后先求平均；曲线为50个场景的均值，阴影为场景级95% Student-t置信区间。纵轴为联合移除后F/F(0)，节点移除1–10个，链路移除1–20条。'}])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--instances',type=int,default=50);p.add_argument('--scale-instances',type=int,default=20)
    p.add_argument('--K',type=int,default=300);p.add_argument('--truth-K',type=int,default=1000)
    p.add_argument('--random-repeats',type=int,default=20);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--force',action='store_true');p.add_argument('--plot-only',action='store_true')
    p.add_argument('--compute-only',action='store_true')
    args=p.parse_args()
    if args.plot_only:plot();return
    if args.K!=scenario.BASE_K:raise ValueError('Baseline K must match experiment_config.BASE_K')
    if args.force and OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    data=compute(args);render(data,args)



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

if __name__=='__main__':main()

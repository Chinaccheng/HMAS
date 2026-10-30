"""E4：静态与自适应节点/链路打击，输出图10与表5。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from UAV.experiment_common import *
from UAV import experiment_config as scenario

def worker(index,args):
    model=model_for(60,index);records=[]
    for kind,methods in [('node',ATTACK_NODE),('edge',ATTACK_EDGE)]:
        for method in methods:
            for repeat in range(args.random_repeats if method=='random' else 1):
                for mode in ['static','adaptive']:
                    if method=='random' and mode=='adaptive':
                        previous=records[-1];records.append({**previous,'mode':mode,'identical_to_random_static':True});continue
                    path=attack_path(model,index,kind,method,mode,repeat,args.K)
                    records.append({'kind':kind,'method':method,'mode':mode,'repeat':repeat,
                                    **path,**robustness_half(path['path'])})
    return {'instance':index,'scene_seed':model.scene_seed,'search_seed':scenario.seeds(60,index)[1],
        'config':model.config.to_dict(),'records':records}

def render(out,data,args):
    repetitions=[];metrics=[];curves=[];grid=np.linspace(0,.5,101)
    for d in data:
        groups=defaultdict(list)
        for r in d['records']:
            repetitions.append({'instance':d['instance'],**{k:r[k] for k in ['kind','method','mode','repeat','R_rob','f_c','f_c_capped','f_c_censored']}})
            groups[(r['kind'],r['method'],r['mode'])].append(r)
        for (kind,method,mode),rs in groups.items():
            metrics.append({'instance':d['instance'],'kind':kind,'method':method,'mode':mode,
                **{field:mean_ci([r[field] for r in rs])[0] for field in ['R_rob','f_c_capped','f_c_censored']}})
            paths=[r['path'] for r in rs]
            for f in grid:
                row={'instance':d['instance'],'kind':kind,'method':method,'mode':mode,'f':float(f)}
                for field in ['Phi','Phi_P','eta']:
                    row[field]=float(np.mean([np.interp(f,[v['f'] for v in p],[v[field] if v[field] is not None else np.nan for v in p]) for p in paths]))
                row['eta']=paths[0][0]['eta']*row['Phi']/row['Phi_P'] if row['Phi_P']>0 else None
                curves.append(row)
    write_csv(out,'metrics_repeats.csv',repetitions);write_csv(out,'metrics_instance.csv',metrics)
    table=summary(metrics,['kind','mode','method'],['R_rob','f_c_capped','f_c_censored']);write_csv(out,'table5_robustness.csv',table)
    write_csv(out,'curves_instance.csv',curves)
    write_csv(out,'curves_summary.csv',summary(curves,['kind','mode','method','f'],['Phi','Phi_P','eta']))
    tests=paired_tests(metrics,['kind','mode'],'R_rob',False);write_csv(out,'wilcoxon.csv',tests)
    if not args.compute_only:plot(out,args)

def main():
    p=parser(__doc__);p.set_defaults(K=300)
    p.add_argument('--task-stats-only',action='store_true')
    args=p.parse_args();out=ROOT/FOLDERS[4]/'result'
    if args.task_stats_only:
        task_failure_statistics(args);return
    if args.plot_only:
        plot(out,args);return
    if args.K!=300:raise ValueError('E4基准固定K=300，与E1的可行任务口径一致')
    out,data=run_jobs(4,[(i,) for i in range(args.instances)],worker,args,
        {'instances':args.instances,'node':ATTACK_NODE,'edge':ATTACK_EDGE,'modes':['static','adaptive'],'interval':[0,.5],
         'cohort':'E1-matched feasible tasks; K300 lambda10 Lmax12'})
    render(out,data,args)
    task_failure_statistics(args)

def conditional_efficiency(curves):
    """Pool searches within a scene; undefined F/P values do not count as zero."""
    baselines={(r['instance'],r['kind'],r['method'],r['mode']):r['eta'] for r in curves if r['f']==0}
    for r in curves:
        eta0=baselines[(r['instance'],r['kind'],r['method'],r['mode'])]
        r['eta']=eta0*r['Phi']/r['Phi_P'] if r['Phi_P']>0 else None
        if r['eta'] is not None:assert 1/6-2e-12<=r['eta']<=1+2e-12
    rows=summary(curves,['kind','mode','method','f'],['eta'])
    stopped=set()
    for r in sorted(rows,key=lambda r:(r['kind'],r['mode'],r['method'],r['f'])):
        key=(r['kind'],r['mode'],r['method'])
        if r['eta_n']<10:stopped.add(key)
        r['drawn']=key not in stopped
    return rows

def plot(out,args):
    write_captions(out)
    curves=read_csv(out/'curves_instance.csv')
    efficiencies=conditional_efficiency(curves)
    write_csv(out,'curves_instance.csv',curves)
    write_csv(out,'efficiency_summary.csv',efficiencies)
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,86/25.4))
    fig.subplots_adjust(left=.08,right=.97,bottom=.29,top=.84,wspace=.27)
    means=summary([r for r in curves if r['mode']=='static'],['kind','method','f'],['Phi'])
    high=max(r['Phi']+(r['Phi_ci95'] or 0) for r in means)
    for col,(kind,methods) in enumerate([('node',ATTACK_NODE),('edge',ATTACK_EDGE)]):
        ax=axs[col]
        selected=[r for r in curves if r['kind']==kind and r['mode']=='static']
        bands(ax,selected,'f','Phi',methods,{'CMLC':'--'} if kind=='edge' else None)
        ax.set(xlim=(0,.5),xticks=[0,.1,.2,.3,.4,.5],ylim=(0,max(1.03,high+.02)),
            xlabel=r'Removed fraction $f_V$' if kind=='node' else r'Removed fraction $f_E$',ylabel='Functional retention Φ')
        ax.axhline(.5,color='#777777',ls='--',lw=.6)
        panel(ax,chr(97+col),'Node attack' if kind=='node' else 'Link attack')
    for methods,x,ncol in [(ATTACK_NODE,.29,3),(ATTACK_EDGE,.76,4)]:
        handles=[Line2D([],[],color=COLOR[m],ls='--' if x>.5 and m=='CMLC' else '-',marker=MARKER.get(m),
                        lw=1.5 if m=='SCFC' else .8,ms=2.5,
                        label=({'h':r'$h_{ij}$','CMLC':'CMLC (ext.)'}.get(m,LABEL[m]) if x>.5 else LABEL[m])) for m in methods]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(x,.012),ncol=ncol,frameon=False,handlelength=1.1,columnspacing=.7,handletextpad=.35)
    export(out,fig,'fig10_attack',axes=list(axs))

def model_for(n,index):
    return scenario.screened_model(n,index,K=300)[0]

def construction(model,index,K=300):
    draws=scenario.search_inputs(model,index,K=K)
    return draws,model.evaluate(random_inputs=draws)

def attack_path(model,index,kind,method,mode='static',repeat=0,K=300):
    draws,base=construction(model,index,K);n=model.config.node_count
    targets=list(range(n)) if kind=='node' else list(model.initial_edges);total=len(targets)
    if method=='random':
        seed=scenario.seeds(n,index)[1]+900000000+repeat*997+(kind=='edge')*100000
        ranking=np.random.default_rng(seed).permutation(total)
    else:ranking=order(values(model,base,kind,method))
    state=NetworkState();remaining=set(range(total));ev=base
    rows=[{'step':0,'f':0.,'F':base.F,'P':base.P,'eta':base.E,'Phi':1. if base.F else None,'Phi_P':1. if base.P else None}]
    actions=[]
    for step in range(1,math.ceil(total*.5)+1):
        if mode=='adaptive' and method!='random':
            if step>1 and method in ['SCFC','CMLC','u','h']:ev=model.evaluate(state,random_inputs=draws)
            score=values(model,ev,kind,method,state)
            j=min(remaining,key=lambda j:(-score[j],j))
        else:j=int(ranking[step-1])
        remaining.remove(j);state=model.attack(state,kind,[targets[j]]);actions.append(targets[j])
        row=evaluate_row(model,state,draws,base);row.update(step=step,f=step/total);rows.append(row)
    return {'path':rows,'actions':actions,'objects_total':total,'baseline_F':base.F,'baseline_P':base.P}


def task_failure_statistics(args):
    """按任务类型输出失效时刻与骤降比例的附录统计表。"""
    out=ROOT/FOLDERS[4]/'result';rows=[]
    for index in range(args.instances):
        model=model_for(60,index);draws,base=construction(model,index,args.K)
        if np.any(base.f_task<=0):raise ValueError(f'Undefined task denominator in scene {index}')
        ranking=order(values(model,base,'node','SCFC'));state=NetworkState()
        ratios=[np.ones(len(model.tasks))]
        for node in ranking[:30]:
            state=model.attack(state,'node',[int(node)])
            ratios.append(model.evaluate(state,random_inputs=draws,collect=False).f_task/base.f_task)
        ratios=np.asarray(ratios)
        assert np.isfinite(ratios).all()
        for k,task in enumerate(model.tasks):
            failed=np.flatnonzero(ratios[:,k]<.5)
            rows.append({'instance':index,'task':k,'type':TASKS[task.task_type],
                'failure_f':float(failed[0]/60) if len(failed) else None,
                'failure_status':'observed' if len(failed) else '>0.5',
                'abrupt':bool(np.any((ratios[:-1,k]>=.9)&(ratios[1:,k]<=.1)))})
        if (index+1)%10==0:print(f'E4 task statistics: {index+1}/{args.instances}',flush=True)
    write_csv(out,'task_failure_instance.csv',rows);table=[]
    for typ in TASKS:
        selected=[r for r in rows if r['type']==typ]
        times=sorted(float('inf') if r['failure_f'] is None else r['failure_f'] for r in selected)
        median=float(np.median(times))
        fractions=[np.mean([r['abrupt'] for r in selected if r['instance']==i]) for i in range(args.instances)]
        mean,ci,n=mean_ci(fractions)
        table.append({'type':typ,'tasks':len(selected),'median_failure_f':median if np.isfinite(median) else '>0.5',
            'censored_tasks':sum(r['failure_f'] is None for r in selected),
            'abrupt_fraction':mean,'abrupt_ci95':ci,'independent_scenes':n})
    write_csv(out,'task_failure_summary.csv',table)
    write_captions(out)


def write_captions(out):
    write_csv(out,'captions.csv',[{'figure':'图10','caption':
        '50个与E1–E3、E5相同的S2可行任务场景；K=300、λ=10、Lmax=12、α=3，侦察/通信/执行半径为30/25/20。'
        '静态策略按初始评分逐个移除，自适应策略每步重算；表5分别汇总两种策略，图中展示静态策略。'
        '随机策略每场景重复20次。(a)节点打击，(b)链路打击；仅展示功能保持率Φ=F(f)/F(0)，横轴分别为节点移除比例f_V与链路移除比例f_E，范围0–0.5。链路策略图例使用h_ij及扩展版本CMLC (ext.)。曲线与阴影分别为50个场景的均值与95% Student-t置信区间，水平虚线为Φ=0.5。'
        '表中f_c_capped是在0.5处截尾的阈值到达比例，未到达比例另列，不能解释为实际失效点；任务补充统计中未到达者记录为>0.5。',
        'suggested_main_text':'本实验中，功能下降几乎全部来自实现率降低，成功搜索的平均实现效率基本保持稳定。'}])

if __name__=='__main__':main()

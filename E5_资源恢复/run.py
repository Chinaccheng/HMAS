"""E5 v11：分批打击、3步平台、逐动作恢复、固定50%预算、完整边际收益与打击恢复矩阵。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from UAV.experiment_common import *
BUDGETS=[.5]
MARGINAL_METHODS=['random','betweenness','loop','CMLC','SCFC','greedy']

def worker(kind,intensity,attack,index,args):
    intensity=float(intensity);model=model_for(60,index);draws,base=construction(model,index,args.K)
    # 预热主模型和批量恢复搜索接口。
    model.scfc(base);state_loops(model)
    model.evaluate_functions([NetworkState()],random_inputs=draws_for(model,index,scenario.GAIN_K,offset=1500000000))
    scores=recovery_scores(model,base)
    damages={};runs=[];marginal_runs=[]
    for method in RECOVERY:
        repeats=args.random_repeats if method=='random' or attack in ['random','area'] else 1
        for repeat in range(repeats):
            damage_repeat=repeat if attack in ['random','area'] else 0
            if damage_repeat not in damages:
                state,events,protocol=damage_path(model,index,kind,attack,intensity,scores,draws,base,damage_repeat)
                damages[damage_repeat]={'state':state,'events':events,'protocol':protocol}
            damage=damages[damage_repeat];state=damage['state']
            log=recover_full(model,index,state,method,scores,draws,base,repeat,budget=.5)
            metrics=[]
            for budget in BUDGETS:
                met,_=recovery_metrics(base,damage['events'],damage['protocol'],log,budget,120)
                met.update(budget=budget)
                metrics.append(met)
            runs.append({'method':method,'repeat':repeat,'damage_repeat':damage_repeat,'actions':log,'metrics':metrics})
            if kind=='node' and intensity==.2 and attack=='SCFC' and method in MARGINAL_METHODS:
                full=recover_full(model,index,state,method,scores,draws,base,repeat,budget=1.)
                assert all(a['target']==b['target'] and a['F']==b['F'] for a,b in zip(log,full))
                assert np.isclose(full[-1]['F'],base.F,atol=2e-14,rtol=0)
                marginal_runs.append({'method':method,'repeat':repeat,'actions':full})
    for damage in damages.values():
        damage['failed_nodes']=sorted(damage['state'].failed_nodes);damage['damaged_edges']=sorted(damage['state'].damaged_edges);del damage['state']
    return {'instance':index,'kind':kind,'intensity':intensity,'attack':attack,'baseline_F':base.F,
            'scene_seed':model.scene_seed,'config':model.config.to_dict(),'manifest':scenario.manifest(model,index,base),
            'damage_cases':damages,'runs':runs,'marginal_runs':marginal_runs}

def render(out,data,args):
    from types import SimpleNamespace
    instance=[];curves=[];gains=[];repeats=[]
    for d in data:
        by_method=defaultdict(list)
        for run in d['runs']:
            by_method[run['method']].append(run)
            for met in run['metrics']:
                repeats.append({'instance':d['instance'],'kind':d['kind'],'intensity':d['intensity'],'attack':d['attack'],
                                'method':run['method'],'repeat':run['repeat'],**met})
        for method,runs in by_method.items():
            for budget in BUDGETS:
                selected=[next(m for m in r['metrics'] if m['budget']==budget) for r in runs]
                row={'instance':d['instance'],'kind':d['kind'],'intensity':d['intensity'],'attack':d['attack'],'method':method,'budget':budget}
                for field in ['R_res','R_F','Phi_final']:
                    row[field]=mean_ci([r[field] for r in selected])[0]
                instance.append(row)
            if d['attack']=='SCFC':
                trajectories=[]
                for run in runs:
                    damage=d['damage_cases'].get(str(run['damage_repeat']),d['damage_cases'].get(run['damage_repeat']))
                    _,phi=recovery_metrics(SimpleNamespace(F=d['baseline_F']),damage['events'],damage['protocol'],run['actions'],.5,120)
                    trajectories.append(phi)
                average=np.mean(trajectories,axis=0)
                curves.extend({'instance':d['instance'],'kind':d['kind'],'intensity':d['intensity'],'method':method,'time':t,'Phi':float(v)} for t,v in enumerate(average))
        for method in MARGINAL_METHODS:
            runs=[r for r in d['marginal_runs'] if r['method']==method]
            if not runs:continue
            total=d['damage_cases'].get('0',d['damage_cases'].get(0))['protocol']['node_damage']
            initial=d['damage_cases'].get('0',d['damage_cases'].get(0))['events'][-1]['F']
            for step in range(total+1):
                vals=[((run['actions'][step-1]['F'] if step else initial)-initial)/d['baseline_F'] for run in runs]
                gains.append({'instance':d['instance'],'method':method,'resource':step/total,'step':step,'gain':float(np.mean(vals))})
    write_csv(out,'scenario_manifest.csv',[next(d['manifest'] for d in data if d['instance']==i) for i in sorted({d['instance'] for d in data})])
    write_csv(out,'metrics_repeats.csv',repeats);write_csv(out,'metrics_instance.csv',instance)
    write_csv(out,'recovery_curves_instance.csv',curves);write_csv(out,'marginal_gains_instance.csv',gains)
    fields=['R_res','R_F','Phi_final']
    table=summary(instance,['kind','intensity','attack','method','budget'],fields);write_csv(out,'table6_all.csv',table)
    for kind in ['node','edge','mixed']:
        selected=[r for r in table if r['kind']==kind and r['attack']=='SCFC' and r['budget']==.5]
        write_csv(out,f'table6_{kind}.csv',selected)
        wide=[]
        for method in RECOVERY:
            row={'method':method}
            for level in [.1,.2,.3]:
                r=next(r for r in selected if r['method']==method and r['intensity']==level)
                for field in fields:
                    row[f'{int(level*100)}pct_{field}']=r[field];row[f'{int(level*100)}pct_{field}_ci95']=r[field+'_ci95']
            wide.append(row)
        write_csv(out,f'table6_{kind}_wide.csv',wide)
    write_csv(out,'recovery_curves_summary.csv',summary(curves,['kind','intensity','method','time'],['Phi']))
    tests=paired_tests([r for r in instance if r['budget']==.5],['kind','intensity','attack'],'R_F',True)
    write_csv(out,'wilcoxon.csv',tests)
    gaps=[]
    for kind in ['node','edge','mixed']:
        for level in [.1,.2,.3]:
            for baseline in ['CMLC','greedy']:
                selected=[r for r in instance if r['kind']==kind and r['intensity']==level and r['attack']=='SCFC' and r['budget']==.5]
                a={r['instance']:r['R_F'] for r in selected if r['method']=='SCFC'};b={r['instance']:r['R_F'] for r in selected if r['method']==baseline}
                diff=np.array([a[i]-b[i] for i in sorted(a)]);mu,ci,n=mean_ci(diff)
                gaps.append({'kind':kind,'intensity':level,'baseline':baseline,'SCFC_minus_baseline':mu,'ci95':ci,
                             'p_two_sided':float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.})
    write_csv(out,'SCFC_recovery_gaps.csv',gaps)
    if not args.compute_only:plot(out,args)

def main():
    args=parser(__doc__).parse_args()
    if args.plot_only:
        plot(ROOT/FOLDERS[5]/'result',args);return
    jobs=[(kind,str(level),'SCFC',i) for kind in ['node','edge','mixed'] for level in [.1,.2,.3] for i in range(args.instances)]
    jobs += [('node','0.2',attack,i) for attack in ['random','area','degree','CMLC'] for i in range(args.instances)]
    out,data=run_jobs(5,jobs,worker,args,{'instances':args.instances,'main_attack':'SCFC','main_kinds':['node','edge','mixed'],
        'levels':[.1,.2,.3],'durations':[3,6,9],'hold_steps':3,'t_a':5,'T':120,'budget':.5,
        'matrix_attacks':['random','area','degree','CMLC','SCFC'],'matrix_scope':'medium node damage',
        'recovery':RECOVERY,'gain_K':scenario.GAIN_K,'gain_draws':'independent batch; same candidates share random inputs',
        'mixed_budget':'one common action budget; node and edge each cost1; blocked edges wait for nodes',
        'marginal_simulation':'separate full recovery for medium SCFC node damage; identical 50% prefix',
        'timing_experiment':False,'greedy_interpretation':'one-step baseline; not an upper bound'})
    render(out,data,args)

def plot(out,args):
    write_captions(out)
    instance=read_csv(out/'metrics_instance.csv');curves=read_csv(out/'recovery_curves_instance.csv')
    gains=read_csv(out/'marginal_gains_instance.csv')
    args.instances=len({r['instance'] for r in instance})
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    from matplotlib.patches import Patch
    fig,ax=plt.subplots(3,3,figsize=(190/25.4,163/25.4));fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.87,wspace=.22,hspace=.38)
    ymax=max(1.06,max(r['Phi'] for r in curves)+.03)
    for i,kind in enumerate(['node','edge','mixed']):
        for j,level in enumerate([.1,.2,.3]):
            rs=[r for r in curves if r['kind']==kind and r['intensity']==level];a=ax[i,j]
            a.axvspan(5,5+{.1:3,.2:6,.3:9}[level],color='#E6E6E6',lw=0,zorder=0,rasterized=True)
            bands(a,rs,'time','Phi',RECOVERY)
            a.axvline(5+{.1:3,.2:6,.3:9}[level]+3,color='#888888',ls=':',lw=.6)
            a.set(xlim=(0,120),ylim=(0,ymax),xticks=[0,40,80,120])
            if i<2:a.tick_params(labelbottom=False)
            else:a.set_xlabel('Time step')
            if j==0:a.set_ylabel('Functional retention Φ')
            panel(a,chr(97+i*3+j),'')
        p=ax[i,0].get_position();fig.text(.025,(p.y0+p.y1)/2,{'node':'Node damage','edge':'Link damage','mixed':'Mixed damage'}[kind],rotation=90,ha='center',va='center',fontsize=9)
    for j,title in enumerate(['Light: 10%','Medium: 20%','Heavy: 30%']):
        p=ax[0,j].get_position();fig.text((p.x0+p.x1)/2,.945,title,ha='center',weight='bold',fontsize=9)
        fig.add_artist(Line2D([p.x0,p.x1],[.932,.932],transform=fig.transFigure,color='#666666',lw=.5))
    legend(fig,RECOVERY,4,.005);export(out,fig,'fig11_recovery',axes=list(ax.flat))
    # Quantitative pair: finite-budget process resilience and complete resource-response curve.
    fig,ax=plt.subplots(1,2,figsize=(190/25.4,96/25.4))
    fig.subplots_adjust(left=.08,right=.98,bottom=.40,top=.87,wspace=.30)
    levels=[.1,.2,.3];level_colors=['#B8CCD9','#789EB9','#3F6B8C']
    plotnode=[r for r in instance if r['kind']=='node' and r['attack']=='SCFC' and r['budget']==.5]
    for j,level in enumerate(levels):
        rr=[r for r in plotnode if r['intensity']==level];mus=[];cis=[]
        for method in RECOVERY:
            mu,ci,_=mean_ci([r['R_F'] for r in rr if r['method']==method]);mus.append(mu);cis.append(ci or 0)
        ax[0].bar(np.arange(7)+(j-1)*.25,mus,width=.23,color=level_colors[j],yerr=cis,capsize=1.5,error_kw={'lw':.6})
    ax[0].set(xticks=range(7),xticklabels=[LABEL[m] for m in RECOVERY],ylabel='Process resilience RF')
    ax[0].tick_params(axis='x',labelrotation=90)
    for tick in ax[0].get_xticklabels():tick.set_rotation_mode('anchor');tick.set_ha('right');tick.set_va('center')
    bands(ax[1],gains,'resource','gain',MARGINAL_METHODS)
    ax[1].set(xlim=(0,1),xlabel='Recovered resource fraction',ylabel='Cumulative functional gain')
    for a,l,t in zip(ax,'ab',['Process resilience','Marginal resource benefit']):panel(a,l,t)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(level_colors,['Light','Medium','Heavy'])],
        loc='lower center',bbox_to_anchor=(.28,.015),ncol=3,frameon=False,columnspacing=.8,handlelength=1)
    handles=[Line2D([],[],color=COLOR[m],ls='--' if m=='greedy' else '-',marker=MARKER.get(m),
        lw=1.6 if m=='SCFC' else .85,ms=2.5,label=LABEL[m]) for m in MARGINAL_METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.75,.015),ncol=3,frameon=False,columnspacing=.8,handlelength=1.2)
    export(out,fig,'fig12_recovery_metrics',axes=list(ax))
    attacks=['random','area','degree','CMLC','SCFC'];matrix=np.array([[mean_ci([r['R_F'] for r in instance if r['kind']=='node' and r['intensity']==.2 and r['attack']==attack and r['method']==method and r['budget']==.5])[0] for method in RECOVERY] for attack in attacks])
    fig,a=plt.subplots(figsize=(190/25.4,86/25.4));fig.subplots_adjust(left=.14,right=.87,bottom=.30,top=.9)
    im=a.imshow(matrix,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto',rasterized=True)
    a.set(xticks=range(7),xticklabels=[LABEL[m] for m in RECOVERY],yticks=range(5),yticklabels=['Random','Regional','Degree','CMLC','SCFC'],xlabel='Recovery strategy',ylabel='Attack strategy')
    a.tick_params(axis='x',labelrotation=30)
    for i in range(5):
        for j in range(7):a.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.85 or matrix[i,j]<.15 else 'black')
    cax=fig.add_axes([.90,.30,.015,.60]);fig.colorbar(im,cax=cax,ticks=[0,.5,1],label='Process resilience RF')
    export(out,fig,'fig13_attack_recovery',axes=[a])

def write_captions(out):
    common='50个独立S2可行任务场景；K=300、λ=10、Lmax=12、α=3。指标类策略使用打击前固定评分；单步贪心每步在同一独立K=300输入批次中比较全部可执行恢复动作，它只优化一步，不是上界。随机恢复以及随机与区域打击每场景重复20次，先在场景内求重复均值，再对场景等权汇总。'
    write_csv(out,'captions.csv',[
        {'figure':'图11','caption':common+'节点、链路和混合受损分别取10%/20%/30%强度，打击持续3/6/9步，结束后保持3步；每步恢复一项动作，预算固定为受损对象数的50%，动作数向下取整。阴影为场景级95% Student-t置信区间。所有恢复策略在恢复开始前共享相同的损伤轨迹。'},
        {'figure':'图12','caption':common+'(a)SCFC静态节点打击下50%预算的过程韧性RF，三根柱分别为轻、中、重度打击；误差棒为50个场景的95% Student-t置信区间。(b)单独仿真中度SCFC静态节点打击后的完整恢复；横轴为已执行动作数/受损对象数，纵轴为Φ(t)−Φ(tr)。(b)展示随机、介数、作战环、CMLC、SCFC与单步贪心，不含度策略。每种策略的前50%动作与(a)同场景序列一致；阴影为场景级95%置信区间。'},
        {'figure':'图13','caption':common+'中度节点受损、50%恢复预算；单元格为过程韧性RF的场景均值，色阶固定0–1；R_F超过1的格子按1着色，格内数值及统计结果保留原值。'},
        {'figure':'表6','caption':'节点、链路和混合受损分别汇总；每种恢复策略对应轻/中/重度打击的R_res、R_F及Φ(T)，均为50%恢复预算。另表给出SCFC减CMLC、SCFC减单步贪心的配对差值及场景级95%置信区间；wilcoxon.csv记录配对检验。'}])


if __name__=='__main__':main()

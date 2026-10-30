"""E7 v11：三规模核心实验、完整单因素扫描和alpha×通信半径矩阵。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from UAV.experiment_common import *
ALPHAS=[0,1,2,3,5,8]
KS=[50,100,200,300,500,1000]

def structural(model,ev):
    return {'edges':len(model.initial_edges),'hyperedges':len(ev.hyperedge_counts),
            'groups_per_task':float(np.mean([len(r) for r in ev.task_combinations])),
            'median_groups':float(np.median([len(r) for r in ev.task_combinations])),
            'J':mean_ci(model.scores(ev).concentration)[0],'F0':ev.F,'P0':ev.P,'eta0':ev.E}

def scale_worker(n,index,args):
    model=model_for(n,index);draws,base=construction(model,index,args.K);old=cached_e2(n,index)
    attacks=[]
    for method in ['random','degree','CMLC','SCFC']:
        for repeat in range(args.random_repeats if method=='random' else 1):
            result=attack_path(model,index,'node',method,'static',repeat,args.K)
            attacks.append({'method':method,'repeat':repeat,**robustness_half(result['path'])})
    model.scfc(base);state_loops(model);model.evaluate_functions([NetworkState()],random_inputs=draws)
    scores=recovery_scores(model,base);state,events,protocol=damage_path(model,index,'node','SCFC',.2,scores,draws,base)
    recovery=[]
    for method in ['random','degree','CMLC','SCFC','greedy']:
        for repeat in range(args.random_repeats if method=='random' else 1):
            log=recover_full(model,index,state,method,scores,draws,base,repeat)
            met,_=recovery_metrics(base,events,protocol,log,.5,model.config.horizon)
            recovery.append({'method':method,'repeat':repeat,**met})
    return {'factor':'scale','value':n,'instance':index,'structure':structural(model,base),
            'accuracy':[r for r in old['metrics'] if r['kind']=='node'],'noise_tau':old['raw']['node']['noise_tau'],
            'attacks':attacks,'recovery':recovery,'manifest':scenario.manifest(model,index,base),'config':model.config.to_dict(),'scene_seed':model.scene_seed}

def worker(factor,value,index,args):
    if factor=='scale':return scale_worker(int(value),index,args)
    if factor=='alpha_rc':
        actual=10000+index
        try:model=model_for(60,actual,communication_radius=float(value))
        except scenario.InfeasibleScenario as exc:return {'factor':factor,'value':value,'instance':index,'status':'non_generatable','reason':str(exc),'rows':[]}
        draws,ev=construction(model,actual,args.K)
        truth,_=true_losses(model,actual,'node',1000,2);true=truth[:,0];rows=[]
        for alpha in ALPHAS:
            score,_=model.scfc(ev,alpha=alpha,kind='node')
            rows.append({'alpha':alpha,'tau':correlation(score,true),'F0':ev.F,'P0':ev.P,'eta0':ev.E})
        return {'factor':factor,'value':float(value),'instance':index,'seed_index':actual,'status':'ok','manifest':scenario.manifest(model,actual,ev),'rows':rows}
    model=model_for(60,index);rows=[]
    if factor=='K':
        reference=model.evaluate(random_inputs=draws_for(model,index,1000));score_ref,_=model.scfc(reference,kind='node')
        for K in KS:
            ev=model.evaluate(random_inputs=draws_for(model,index,K));score,_=model.scfc(ev,kind='node')
            fs=[model.evaluate(random_inputs=draws_for(model,index,K,repeat),collect=False).F for repeat in range(10)]
            rows.append({'K':K,'F_cv':float(np.std(fs,ddof=1)/np.mean(fs)) if np.mean(fs)>0 else None,
                         'rank_stability':correlation(score,score_ref),'F_repeats':fs,'F0':ev.F,'P0':ev.P,'eta0':ev.E})
    elif factor=='alpha':
        old=cached_e2(60,index);draws,ev=construction(model,index,args.K)
        for alpha in ALPHAS:
            a,b=model.scfc(ev,alpha=alpha);b=np.array([b[e] for e in model.initial_edges])
            rows.append({'alpha':alpha,'node_tau':correlation(a,np.array(old['raw']['node']['truth_losses'])[:,0]),
                'edge_tau':correlation(b,np.array(old['raw']['edge']['truth_losses'])[:,0]),
                'CMLC_tau':next(r['tau'] for r in old['metrics'] if r['kind']=='node' and r['method']=='CMLC'),
                'F0':ev.F,'P0':ev.P,'eta0':ev.E})
    else:
        field={'lambda':'priority_weight','sensing':'sensing_radius','execution':'execution_radius','hops':'max_hops'}[factor]
        val=int(value) if factor=='hops' else float(value);
        try:model=model_for(60,index,**{field:val})
        except scenario.InfeasibleScenario as exc:return {'factor':factor,'value':value,'instance':index,'status':'non_generatable','reason':str(exc),'rows':[]}
        draws,ev=construction(model,index,args.K)
        true,bases=true_losses(model,index,'node',1000,2);score,_=model.scfc(ev,kind='node')
        rows=[{'parameter':float(value),'tau':correlation(score,true[:,0]),'noise_tau':correlation(true[:,0],true[:,1]),
               'F0':ev.F,'P0':ev.P,'eta0':ev.E,
               'mean_team_size':float(np.mean([len(g) for gg in ev.task_combinations for g in gg])),
               'truth':true,'SCFC':score,'truth_baselines':bases}]
    return {'factor':factor,'value':value,'instance':index,'status':'ok','manifest':scenario.manifest(model,index),'scene_seed':model.scene_seed,'config':model.config.to_dict(),'rows':rows}

def render(out,data,args):
    write_csv(out,'scenario_status.csv',[{k:r[k] for k in ['factor','value','instance','status','reason'] if k in r} for r in data])
    write_csv(out,'scenario_manifest.csv',[{'factor':r['factor'],'value':r['value'],**r['manifest']} for r in data if 'manifest' in r])
    scale=[]
    for d in data:
        if d['factor']!='scale':continue
        n=int(d['value']);acc={r['method']:r for r in d['accuracy']}
        for method in ['random','degree','betweenness','loop','CMLC','SCFC','greedy']:
            scale.append({'n':n,'instance':d['instance'],'method':method,'tau':acc.get(method,{}).get('tau'),
                'noise_tau':d['noise_tau'],'R_rob':mean_ci([r['R_rob'] for r in d['attacks'] if r['method']==method])[0],
                'R_F':mean_ci([r['R_F'] for r in d['recovery'] if r['method']==method])[0],**d['structure']})
    for index in range(args.instances):
        acc_rows=[r for r in read_csv(ROOT/'E2_识别准确性/result/metrics_raw.csv') if r['n']==60 and r['instance']==index]
        acc={'metrics':acc_rows,'raw':{'node':{'noise_tau':next(r['noise_tau'] for r in acc_rows if r['kind']=='node')}}}
        att={'records':[r for r in read_csv(ROOT/'E4_定向打击/result/metrics_instance.csv') if r['instance']==index]}
        rec={'runs':[{'method':r['method'],'metrics':[r]} for r in read_csv(ROOT/'E5_资源恢复/result/metrics_instance.csv') if r['instance']==index and r['kind']=='node' and r['intensity']==.2 and r['attack']=='SCFC' and r['budget']==.5]}
        model=model_for(60,index);_,ev=construction(model,index,args.K);structure=structural(model,ev)
        a={r['method']:r for r in acc['metrics'] if r['kind']=='node'}
        for method in ['random','degree','betweenness','loop','CMLC','SCFC','greedy']:
            scale.append({'n':60,'instance':index,'method':method,'tau':a.get(method,{}).get('tau'),
                'noise_tau':acc['raw']['node']['noise_tau'],
                'R_rob':mean_ci([r['R_rob'] for r in att['records'] if r['kind']=='node' and r['mode']=='static' and r['method']==method])[0],
                'R_F':mean_ci([m['R_F'] for r in rec['runs'] if r['method']==method for m in r['metrics'] if m['budget']==.5])[0],**structure})
    write_csv(out,'scale_instance.csv',scale)
    ss=summary(scale,['n','method'],['tau','noise_tau','R_rob','R_F','edges','hyperedges','groups_per_task','median_groups','J','F0','P0','eta0'])
    write_csv(out,'scale_summary.csv',ss)
    table=[]
    for n in [30,60,120]:
        r=next(r for r in ss if r['n']==n and r['method']=='SCFC');row={'n':n}
        for field in ['edges','hyperedges','groups_per_task','median_groups','J','F0','P0','eta0','R_rob','R_F']:
            row[field]=r[field];row[field+'_ci95']=r[field+'_ci95']
        for method in ['SCFC','CMLC','loop']:
            r=next(r for r in ss if r['n']==n and r['method']==method);row[method+'_tau']=r['tau'];row[method+'_tau_ci95']=r['tau_ci95']
        table.append(row)
    write_csv(out,'table7_scale.csv',table)
    params=[]
    for d in data:
        if d['factor']=='scale':continue
        for r in d['rows']:
            clean={k:v for k,v in r.items() if k not in ['truth','SCFC','F_repeats']}
            clean.update(factor=d['factor'],value=d['value'],instance=d['instance'])
            params.append(clean)
    write_csv(out,'parameter_instance.csv',params)
    function_summary(out,params)
    if not args.compute_only:plot(out,args)

def function_summary(out,params):
    rows=[{'factor':r['factor'],
        'parameter':next(r[k] for k in ['K','alpha','parameter'] if r.get(k) is not None),
        'communication_radius':r['value'] if r['factor']=='alpha_rc' else None,
        **{k:r[k] for k in ['F0','P0','eta0']}} for r in params]
    write_csv(out,'parameter_function_summary.csv',summary(rows,
        ['factor','parameter','communication_radius'],['F0','P0','eta0']))

def main():
    args=parser(__doc__).parse_args()
    if args.plot_only:
        plot(ROOT/FOLDERS[7]/'result',args);return
    jobs=[('scale',str(n),i) for n in [30,120] for i in range(20)]
    jobs += [(factor,'all',i) for factor in ['K','alpha'] for i in range(10)]
    settings=[('lambda',str(v)) for v in [3,5,10,15,20]]
    settings += [('sensing',str(v)) for v in [27,30,33,36,39]]+[('execution',str(v)) for v in [14,16,18,20,22,24]]
    settings += [('hops',str(v)) for v in [8,10,12,16,20]]+[('alpha_rc',str(v)) for v in [21,23,25,27,29]]
    jobs += [(factor,value,i) for factor,value in settings for i in range(10)]
    out,data=run_jobs(7,jobs,worker,args,{'scale_instances':20,'base_scale_from':['E2','E4','E5'],
        'sensitivity_instances':10,'K':KS,'alpha':ALPHAS,'settings':settings,'K_cv':'10 independent searches within each fixed scene',
        'truth_K':1000,'scale_truth_K':500,'K_random_pairing':'first21 hop inputs identical to E2; additional hops use independent extension',
        'factorial_cohort':'E6 independent radius cohort first10 scenes; paired across radii','timing_experiment':False})
    render(out,data,args)

def plot(out,args):
    write_captions(out)
    ss=read_csv(out/'scale_summary.csv')
    params=read_csv(out/'parameter_instance.csv')
    plt=plotting();from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator,NullFormatter
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig,ax=plt.subplots(1,3,figsize=(190/25.4,90/25.4));fig.subplots_adjust(left=.07,right=.98,bottom=.34,top=.82,wspace=.75)
    specs=[('tau',['degree','betweenness','loop','CMLC','SCFC'],'Node Kendall τb'),
           ('R_rob',['random','degree','CMLC','SCFC'],'Robustness Rrob'),
           ('R_F',['random','degree','CMLC','SCFC','greedy'],'Process resilience RF')]
    for a,(field,methods,ylabel) in zip(ax[:3],specs):
        for method in methods:
            rs=sorted([r for r in ss if r['method']==method],key=lambda r:r['n'])
            a.errorbar([r['n'] for r in rs],[r[field] for r in rs],yerr=[r[field+'_ci95'] for r in rs],
                       color=COLOR[method],ls='--' if method=='greedy' else '-',marker=MARKER.get(method),ms=3,
                       lw=1.5 if method=='SCFC' else .8,capsize=1.5)
        a.set(xticks=[30,60,120],xlabel='Nodes N',ylabel=ylabel)
        a.set_xscale('log',base=2)
        a.xaxis.set_major_locator(FixedLocator([30,60,120]))
        a.xaxis.set_major_formatter(FixedFormatter(['30','60','120']))
        a.xaxis.set_minor_locator(NullLocator());a.xaxis.set_minor_formatter(NullFormatter())
    rs=sorted([r for r in ss if r['method']=='SCFC'],key=lambda r:r['n'])
    ax[0].plot([r['n'] for r in rs],[r['noise_tau'] for r in rs],color='#777777',ls='--',lw=.8)
    for a,l,t in zip(ax,'abc',['Identification','Attack','Recovery']):panel(a,l,t)
    handles=[Line2D([],[],color=COLOR[m],ls='--' if m=='greedy' else '-',marker=MARKER.get(m),ms=3,lw=1,label=LABEL[m]) for m in ['random','degree','betweenness','loop','CMLC','SCFC','greedy']]
    handles += [Line2D([],[],color='#777777',ls='--',lw=.8,label='Truth repeatability')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.005),ncol=3,frameon=False,columnspacing=1.1,handlelength=1.4)
    export(out,fig,'fig16_scale',axes=list(ax))
    fig,ax=plt.subplots(2,3,figsize=(190/25.4,137/25.4));fig.subplots_adjust(left=.10,right=.90,bottom=.12,top=.92,wspace=.65,hspace=.55)
    def line(a,factor,xfield,yfield,color,style='-',marker='o',right=False,exclude=()):
        rs=[r for r in params if r['factor']==factor and float(r[xfield]) not in exclude];xs=sorted({float(r[xfield]) for r in rs});ys=[];cs=[]
        for xx in xs:
            m,c,n=mean_ci([r[yfield] for r in rs if float(r[xfield])==xx]);ys.append(m);cs.append(c or 0)
        a.errorbar(xs,ys,yerr=cs,color=color,ls=style,marker=marker,ms=2.5,lw=.75 if right else 1,capsize=1.5)
        return xs,ys,cs
    twinK=ax[0,0].twinx();twinK.spines['right'].set_visible(True)
    line(ax[0,0],'K','K','F_cv',COLOR['degree']);line(twinK,'K','K','rank_stability',COLOR['SCFC'],'--',right=True,exclude=(1000,))
    ax[0,0].set(xscale='log',xticks=[50,300,1000],xticklabels=['50','300','1000'],xlabel='Searches K',ylabel='Function CV');ax[0,0].yaxis.label.set_color(COLOR['degree']);ax[0,0].tick_params(axis='y',colors=COLOR['degree'])
    twinK.set_ylabel('Rank stability τ',color=COLOR['SCFC'],labelpad=2);twinK.tick_params(length=2,pad=2,colors=COLOR['SCFC']);ax[0,0].axvline(300,color='#888888',ls=':',lw=.6)
    line(ax[0,1],'alpha','alpha','node_tau',COLOR['SCFC']);line(ax[0,1],'alpha','alpha','edge_tau',COLOR['degree'],'--',marker='s')
    cm=mean_ci([r['CMLC_tau'] for r in params if r['factor']=='alpha' and r['alpha']==3]);ax[0,1].axhline(cm[0],color=COLOR['CMLC'],lw=.8)
    ax[0,1].axhspan(cm[0]-cm[1],cm[0]+cm[1],color=COLOR['CMLC'],alpha=.08,lw=0,rasterized=True)
    ax[0,1].set(xlabel='Locality α',ylabel='Kendall τb',xticks=ALPHAS,ylim=(-.05,1.02))
    ax[0,1].legend(handles=[Line2D([],[],color=COLOR['SCFC'],marker='o',ms=2.5,lw=1,label='Node'),
        Line2D([],[],color=COLOR['degree'],marker='s',ms=2.5,ls='--',lw=1,label='Link'),
        Line2D([],[],color=COLOR['CMLC'],lw=.8,label='CMLC (node)')],
        loc='lower left',ncol=2,fontsize=7.5,frameon=False,columnspacing=.5,handlelength=1.1,handletextpad=.4,labelspacing=.25,borderaxespad=.3)
    line(ax[0,2],'lambda','parameter','tau',COLOR['SCFC'])
    twinL=ax[0,2].twinx();twinL.spines['right'].set_visible(True)
    line(twinL,'lambda','parameter','P0','#777777',right=True)
    twinL.set_ylabel('Search success P(0)',color='#777777',labelpad=2);twinL.tick_params(length=2,pad=2,colors='#777777')
    ax[0,2].set(xlabel='Search bias λ',ylabel='Node Kendall τb',xticks=[3,5,10,15,20]);ax[0,2].yaxis.label.set_color(COLOR['SCFC']);ax[0,2].tick_params(axis='y',colors=COLOR['SCFC']);ax[0,2].axvline(10,color='#888888',ls=':',lw=.6)
    line(ax[1,0],'sensing','parameter','tau',COLOR['SCFC']);line(ax[1,0],'execution','parameter','tau',COLOR['degree'],'--',marker='s')
    ax[1,0].axvline(25,color='#888888',ls=':',lw=.6);ax[1,0].set(xlabel='Sensing / execution radius',ylabel='Node Kendall τb',xticks=[14,20,25,30,39])
    lo,hi=ax[1,0].get_ylim();ax[1,0].set_ylim(lo-.6*(hi-lo),hi)
    ax[1,0].legend(handles=[Line2D([],[],color=COLOR['SCFC'],marker='o',ms=2.5,lw=1,label=r'$r_S$ varied'),
        Line2D([],[],color=COLOR['degree'],marker='s',ms=2.5,ls='--',lw=1,label=r'$r_I$ varied')],
        loc='lower right',fontsize=7.5,frameon=False,handlelength=1.2,handletextpad=.4,labelspacing=.25)
    twinH=ax[1,1].twinx();twinH.spines['right'].set_visible(True)
    line(ax[1,1],'hops','parameter','tau',COLOR['SCFC']);line(twinH,'hops','parameter','mean_team_size','#777777',right=True)
    ax[1,1].axvline(12,color='#888888',ls=':',lw=.6);ax[1,1].set(xlabel='Maximum hops',ylabel='Node Kendall τb',xticks=[8,10,12,16,20]);twinH.set_ylabel('Mean team size',color='#777777',labelpad=2);twinH.tick_params(length=2,pad=2,colors='#777777');ax[1,1].yaxis.label.set_color(COLOR['SCFC']);ax[1,1].tick_params(axis='y',colors=COLOR['SCFC'])
    heat=np.array([[mean_ci([r['tau'] for r in params if r['factor']=='alpha_rc' and float(r['value'])==radius and r['alpha']==alpha])[0] for alpha in ALPHAS] for radius in [21,23,25,27,29]])
    im=ax[1,2].imshow(heat,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,rasterized=True)
    ax[1,2].set(xticks=range(6),xticklabels=ALPHAS,yticks=range(5),yticklabels=[21,23,25,27,29],xlabel='Locality α',ylabel='Communication radius')
    for i in range(5):
        for j in range(6):ax[1,2].text(j,i,f'{heat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if heat[i,j]>.85 else 'black')
    pos=ax[1,2].get_position();cax=fig.add_axes([.94,pos.y0,.012,pos.height]);fig.colorbar(im,cax=cax,ticks=[0,.5,1])
    cax.set_title(r'$\tau_b$',fontsize=8,pad=4)
    for a,l,t in zip(ax.flat,'abcdef',['Search count','Locality','Search bias','Coverage radii','Search horizon','Locality × radius']):panel(a,l,t)
    export(out,fig,'fig17_sensitivity',axes=list(ax.flat))

def write_captions(out):
    write_csv(out,'captions.csv',[
        {'figure':'图16','caption':'S1与S3各20个场景，S2为与E1–E5相同的50个场景；等密度缩放，所有条件K=300、λ=10、Lmax=12、α=3，任务位置经可行性拒绝采样。节点排序准确性直接使用E2同一批数据；S2打击和恢复分别使用E4/E5同一批数据。(c)三个规模的过程韧性均在[t_a,120]上计算，统一取T=120，t_a=5；中度节点受损、SCFC静态打击、恢复预算50%。误差棒为场景级95%置信区间；S2真值K′=1000，S1与S3按设计取K′=500；不做耗时实验。'},
        {'figure':'表7','caption':'三个规模统一取 T = 120。R_F在[t_a,T]上按梯形积分除以T−t_a计算，t_a=5；恢复条件为中度节点受损、SCFC静态打击、50%预算。R_F列为SCFC恢复结果，±为独立场景均值的95%置信区间半宽；S1/S3各20个场景，S2为50个场景。'},
        {'figure':'图17','caption':'每个条件10个固定场景，仅改变指定参数；必要时在该条件下对任务位置重新做拒绝采样，网络不重抽，场景与任务指纹见scenario_manifest.csv。主搜索K=300，λ=10，Lmax=12，α=3；真值K′=1000。(a)每场景10次独立搜索得到F(0)变异系数，排序稳定性与同场景K=1000结果比较；该参照点本身不在稳定性曲线上绘制。双纵轴的标签及刻度颜色对应各自曲线；其余面板准确性均用与该条件对应的第一批真值。(c)右轴为当前条件下全部可行任务的搜索成功率。(e)右轴为成功有效超边的平均团队规模。(f)与E6相同的独立通信半径队列前10个场景，色标标题τ_b，色阶0–1；各均值和95%区间按独立场景汇总。'}])


if __name__=='__main__':main()

"""E6 v11：21–29通信半径、三种任务构成、完整角色网格。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from UAV.experiment_common import *

def worker(factor,value,index,args):
    actual_index=10000+index if factor=='radius' else 20000+index if factor=='roles' else index
    changes={}
    if factor=='radius':changes['communication_radius']=float(value)
    if factor=='roles':
        s,d=map(int,value.split('-'));changes.update(n_s=s,n_d=d,n_i=60-s-d)
    composition=value if factor=='composition' else 'uniform'
    try:model=model_for(60,actual_index,task_composition=composition,**changes)
    except scenario.InfeasibleScenario as exc:
        return {'factor':factor,'value':value,'instance':index,'seed_index':actual_index,
            'status':'non_generatable','reason':str(exc),'n_s':changes.get('n_s',24),
            'n_d':changes.get('n_d',12),'n_i':changes.get('n_i',24)}
    draws,ev=construction(model,actual_index,args.K)
    row={'status':'ok','manifest':scenario.manifest(model,actual_index,ev),'factor':factor,'value':value,'instance':index,'seed_index':actual_index,'scene_seed':model.scene_seed,
        'config':model.config.to_dict(),'F0':ev.F,'P0':ev.P,'eta0':ev.E,
        'n_s':model.config.n_s,'n_d':model.config.n_d,'n_i':model.config.n_i}
    s=model.scores(ev);row['median_teams']=float(np.median([len(g) for g in ev.task_combinations]));row['J']=mean_ci(s.concentration)[0];row['J_defined_tasks']=int(np.sum(np.isfinite(s.concentration)))
    if factor=='radius':
        truth,bases=true_losses(model,actual_index,'node',1000,2)
        scores={method:values(model,ev,'node',method) for method in ['SCFC','no_reroute','alpha0','no_failed','degree','CMLC','loop']}
        row.update(truth=truth,truth_baselines=bases,scores=scores,
                   taus={m:correlation(v,truth[:,0]) for m,v in scores.items()},roles=model.roles,
                   positions=model.positions,noise_tau=correlation(truth[:,0],truth[:,1]))
    else:
        methods=['SCFC'] if factor=='roles' else ['random','degree','CMLC','SCFC']
        records=[]
        for method in methods:
            reps=args.random_repeats if method=='random' else 1
            for repeat in range(reps):
                # 构成改变只影响任务要求，不改变几何及原始编号。
                path=attack_path(model,actual_index,'node',method,'static',repeat,args.K)
                records.append({'method':method,'repeat':repeat,**robustness_half(path['path'])})
        row['attacks']=records
    return row

def render(out,data,args):
    write_csv(out,'scenario_status.csv',[{k:r[k] for k in ['factor','value','instance','seed_index','status','reason'] if k in r} for r in data])
    write_csv(out,'scenario_manifest.csv',[{'factor':r['factor'],'value':r['value'],**r['manifest']} for r in data if r['status']=='ok'])
    data=[r for r in data if r['status']=='ok']
    radius=[r for r in data if r['factor']=='radius'];raw=[]
    for r in radius:
        for method,tau in r['taus'].items():raw.append({'instance':r['instance'],'seed_index':r['seed_index'],'radius':float(r['value']),
            'method':method,'tau':tau,'median_teams':r['median_teams'],'J':r['J'],'F0':r['F0'],'P0':r['P0'],'eta0':r['eta0'],
            'gain_CMLC':r['taus']['SCFC']-r['taus']['CMLC'],'gain_loop':r['taus']['SCFC']-r['taus']['loop']})
    write_csv(out,'radius_accuracy_raw.csv',raw)
    write_csv(out,'radius_summary.csv',summary(raw,['radius','method'],['tau','median_teams','J','F0','P0','eta0','gain_CMLC','gain_loop']))
    attacks=[];roles=[]
    for r in data:
        if r['factor']=='composition':
            for method in ['random','degree','CMLC','SCFC']:
                selected=[v for v in r['attacks'] if v['method']==method]
                attacks.append({'instance':r['instance'],'composition':r['value'],'method':method,
                    'R_rob':mean_ci([v['R_rob'] for v in selected])[0]})
        elif r['factor']=='roles':
            roles.append({'instance':r['instance'],'n_s':r['n_s'],'n_d':r['n_d'],'n_i':r['n_i'],
                'pS':r['n_s']/60,'pD':r['n_d']/60,'F0':r['F0'],'P0':r['P0'],'J':r['J'],'R_rob':r['attacks'][0]['R_rob']})
    write_csv(out,'composition_instance.csv',attacks);write_csv(out,'role_instance.csv',roles)
    write_csv(out,'composition_summary.csv',summary(attacks,['composition','method'],['R_rob']))
    role_summary=summary(roles,['n_s','n_d','n_i','pS','pD'],['F0','R_rob','P0','J']);write_csv(out,'role_summary.csv',role_summary)
    write_csv(out,'wilcoxon.csv',paired_tests(attacks,['composition'],'R_rob',False))
    if not args.compute_only:plot(out,args)

def main():
    args=parser(__doc__).parse_args()
    if args.plot_only:
        plot(ROOT/FOLDERS[6]/'result',args);return
    jobs=[('radius',str(r),i) for r in [21,23,25,27,29] for i in range(20)]
    jobs += [('composition',v,i) for v in ['basic','uniform','composite'] for i in range(args.instances)]
    grid=[(6*s,6*d) for s in range(1,9) for d in range(1,10-s)]+[(20,20)]
    jobs += [('roles',f'{s}-{d}',i) for s,d in grid for i in range(args.instances)]
    out,data=run_jobs(6,jobs,worker,args,{'radius_instances':20,'other_instances':args.instances,'radius':[21,23,25,27,29],
         'radius_cohort_offset':10000,'roles_cohort_offset':20000,'role_grid':grid,'true_K':1000,'truth_batches':2})
    render(out,data,args)

def plot(out,args):
    write_captions(out)
    raw=read_csv(out/'radius_accuracy_raw.csv');attacks=read_csv(out/'composition_instance.csv')
    role_summary=read_csv(out/'role_summary.csv')
    radius=[{'value':r['radius'],'median_teams':r['median_teams'],'F0':r['F0']} for r in raw if r['method']=='SCFC']
    plt=plotting();from matplotlib.lines import Line2D
    plt.rcParams['mathtext.fontset'] = 'stix'
    fig,ax=plt.subplots(1,3,figsize=(190/25.4,92/25.4));fig.subplots_adjust(left=.085,right=.97,bottom=.32,top=.82,wspace=.65)
    twin=ax[0].twinx();twin.spines['right'].set_visible(True)
    xs=[21,23,25,27,29];jm=[];jc=[];fm=[];fc=[]
    for r in xs:
        selected=[v for v in radius if float(v['value'])==r]
        m,c,_=mean_ci([v['median_teams'] for v in selected]);jm.append(m);jc.append(c)
        m,c,_=mean_ci([v['F0'] for v in selected]);fm.append(m);fc.append(c)
    ax[0].errorbar(xs,jm,yerr=jc,color=COLOR['degree'],marker='o',ms=3,lw=1,capsize=2)
    twin.errorbar(xs,fm,yerr=fc,color='#777777',lw=.65,marker='s',ms=2,capsize=2)
    ax[0].set(xlabel='Communication radius',ylabel='Median teams per task',xticks=xs)
    twin.set_ylabel('F(0)',color='#777777',labelpad=2);twin.tick_params(length=2,pad=2)
    for method,field,style in [('CMLC','gain_CMLC','-'),('loop','gain_loop','--')]:
        ms=[];cs=[]
        for r in xs:
            selected=[v for v in raw if v['radius']==r and v['method']=='SCFC']
            m,c,_=mean_ci([v[field] for v in selected]);ms.append(m);cs.append(c)
        ax[1].errorbar(xs,ms,yerr=cs,color=COLOR[method],ls=style,marker=MARKER[method],ms=3,lw=1,capsize=2)
    ax[1].set(xlabel='Communication radius',ylabel='SCFC advantage Δτ',xticks=xs)
    for j,method in enumerate(['random','degree','CMLC','SCFC']):
        mus=[];cis=[]
        for value in ['basic','uniform','composite']:
            m,c,_=mean_ci([r['R_rob'] for r in attacks if r['composition']==value and r['method']==method]);mus.append(m);cis.append(c)
        ax[2].bar(np.arange(3)+(j-1.5)*.19,mus,width=.18,color=COLOR[method],yerr=cis,capsize=1.5,error_kw={'lw':.6},zorder=2)
    ax[2].set(xticks=range(3),xticklabels=['Basic','Uniform','Composite'],ylabel='Robustness Rrob',ylim=(0,1.05))
    ax[2].tick_params(axis='x',labelrotation=30)
    for a,letter,title in zip(ax,'abc',['Team redundancy','SCFC advantage','Task composition']):panel(a,letter,title)
    handles=[Line2D([],[],color=COLOR['degree'],marker='o',lw=1,label='Median teams'),Line2D([],[],color='#777777',marker='s',lw=.65,label='F(0)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.19,.035),frameon=False,ncol=1)
    handles=[Line2D([],[],color=COLOR[m],ls=style,lw=1,label='SCFC − '+LABEL[m]) for m,style in [('CMLC','-'),('loop','--')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.035),frameon=False,ncol=1)
    handles=[Line2D([],[],color=COLOR[m],lw=4,label=LABEL[m]) for m in ['random','degree','CMLC','SCFC']]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.83,.035),frameon=False,ncol=2,columnspacing=.7,handlelength=1)
    export(out,fig,'fig14_structure',axes=list(ax))
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,88/25.4));fig.subplots_adjust(left=.075,right=.89,bottom=.23,top=.82,wspace=.40)
    x=np.arange(.1,.9,.1);y=x.copy()
    role_counts=defaultdict(lambda:[0,0])
    for r in read_csv(out/'scenario_status.csv'):
        if r['factor']=='roles':
            ss,dd=map(int,r['value'].split('-'));role_counts[(ss,dd)][r['status']=='ok']+=1
    for a,field,title in zip(axs,['F0','R_rob'],['Initial function','SCFC attack robustness']):
        matrix=np.full((8,8),np.nan)
        for r in role_summary:
            if r['n_s']==20:continue
            matrix[int(round(r['pD']*10))-1,int(round(r['pS']*10))-1]=r[field] if r[field] is not None else np.nan
        im=a.pcolormesh(np.arange(.05,.91,.1),np.arange(.05,.91,.1),np.ma.masked_invalid(matrix),cmap='RdYlGn',vmin=0,vmax=1,shading='flat',edgecolors='white',lw=.5,rasterized=True)
        for (ss,dd),counts in role_counts.items():
            if ss!=20 and counts[1]==0:a.text(ss/60,dd/60,'×',ha='center',va='center',fontsize=8,color='#555555')
        a.scatter([.4],[.2],marker='*',s=60,facecolor='white',edgecolor='black',lw=.7,zorder=4)
        a.scatter([1/3],[1/3],marker='D',s=25,facecolor='white',edgecolor='black',lw=.7,zorder=4)
        a.set(xlim=(.05,.9),ylim=(.05,.9),xlabel='Sensing proportion pS',ylabel='Decision proportion pD',xticks=[.1,.3,.5,.7,.9],yticks=[.1,.3,.5,.7,.9]);a.set_aspect('equal')
        panel(a,'a' if field=='F0' else 'b',title)
        pos=a.get_position();cbax=fig.add_axes([pos.x1+.012,pos.y0,.012,pos.height]);cb=fig.colorbar(im,cax=cbax,ticks=[0,.5,1])
        cb.ax.set_title(r'$F(0)$' if field=='F0' else r'$R_{\mathrm{rob}}$',fontsize=8,pad=6)
    handles=[Line2D([],[],ls='',marker='*',mfc='white',mec='black',ms=7,label='Baseline 2:1:2'),Line2D([],[],ls='',marker='D',mfc='white',mec='black',ms=4,label='Balanced 1:1:1')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
    export(out,fig,'fig15_roles',axes=list(axs))

def write_captions(out):
    status=read_csv(out/'scenario_status.csv');bad=[r for r in status if r['status']!='ok']
    roles=read_csv(out/'role_summary.csv')
    for row in roles:
        cases=[r for r in status if r['factor']=='roles' and r['value']==f"{row['n_s']}-{row['n_d']}"]
        row.update(attempted_networks=len(cases),non_generatable_networks=sum(r['status']!='ok' for r in cases),
            generatable_fraction=sum(r['status']=='ok' for r in cases)/len(cases))
    write_csv(out,'role_summary.csv',roles)
    common='全部条件使用统一K=300、λ=10、Lmax=12、α=3；在每个条件下仅对任务位置做可行性拒绝采样，固定网络与任务要求。'
    write_csv(out,'captions.csv',[
        {'figure':'图14','caption':common+'通信半径每个取值20个独立网络；(a)先在每场景的50个任务内取不同团队数中位数，再画场景均值和95% Student-t置信区间，右轴为F(0)。(b)SCFC与对照指标在同一场景、同一超图和同一批K′=1000真值上求τ差后汇总。(c)每种构成50个场景，改变任务类型后重新检查位置可行性；随机打击每场景重复20次后先求平均。'},
        {'figure':'图15','caption':common+f'每个角色组合固定尝试50个网络；共{len([r for r in bad if r["factor"]=="roles"])}个固定网络无法生成所需的全可行任务库，未替换网络、未填零；原因与编号见scenario_status.csv，有效样本数见role_summary.csv的F0_n/R_rob_n。颜色为可生成场景的均值；无有效样本的组合用×标记。热图范围固定0–1，三元网格之外为空白。'}])


if __name__=='__main__':main()

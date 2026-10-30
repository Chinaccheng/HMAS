from common import *

def A():
    names={1:'table3_task_types table3_scenarios independent_validation_summary',2:'accuracy_summary topk_summary wilcoxon scale_stability table4_evaluation',3:'ablation_summary role_loss_summary top10_role_summary wilcoxon',4:'table5_robustness curves_summary efficiency_summary task_failure_summary wilcoxon',5:'table6_all wilcoxon SCFC_recovery_gaps recovery_curves_summary',6:'radius_summary composition_summary role_summary scenario_status wilcoxon',7:'scale_summary table7_scale parameter_function_summary'}
    manifest=[]
    for e,files in names.items():
        for name in files.split():
            src=DIRS[e]/(name+'.csv');dst=OUT/'A_existing'/f'E{e}'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
            if src.exists():
                shutil.copyfile(src,dst)
                manifest.append({'experiment':e,'file':src.name,'status':'copied_byte_identical','sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
            else:
                manifest.append({'experiment':e,'file':src.name,'status':'missing_original; derived separately','sha256':''})
    save('A_copy_manifest.csv',manifest)
    for e,src,name,keys,metrics in [(4,'curves_instance.csv','A_derived_E4_curves_summary.csv',['kind','mode','method','f'],['Phi','Phi_P','eta']),(5,'recovery_curves_instance.csv','A_derived_E5_recovery_curves_summary.csv',['kind','intensity','method','time'],['Phi'])]:
        save(name,summary(read(e,src),keys,metrics))

def B01():
    rows=[r for r in read(5,'metrics_instance.csv') if r['attack']=='SCFC' and r['budget']==.5]
    save('B01_recovery_edge_mixed.csv',summary([r for r in rows if r['kind'] in ('edge','mixed')],['kind','intensity','method'],['R_res','R_F','Phi_final']))
    tests=[]
    for kind in ['node','edge','mixed']:
        for level in [.1,.2,.3]:
            rs=[r for r in rows if r['kind']==kind and r['intensity']==level]
            for metric in ['R_F','Phi_final']:
                best=best_method(rs,metric)
                for label,b in ([('greedy','greedy')] if kind=='node' else [('best_indicator',best),('CMLC','CMLC'),('greedy','greedy')]):
                    tests.append({**compare(rs,metric,b,kind=kind,intensity=level),'selection':label})
    save('B01_tests.csv',tests)

def B05():
    rows=[]
    for r in read(2,'topk_summary.csv'):
        if (r['kind']=='node' and r['k'] in (5,10)) or (r['kind']=='edge' and r['k'] in (10,20)):
            rows.append({k:v for k,v in r.items() if k not in ['ci95','instances'] }|{'Phi_ci95':r['ci95'],'Phi_n':r['instances']})
    save('B05_topk_values.csv',rows)
    raw=read(5,'marginal_gains_instance.csv');out=[];instance=[]
    for (i,m) in sorted({(r['instance'],r['method']) for r in raw}):
        rs=sorted([r for r in raw if r['instance']==i and r['method']==m],key=lambda r:r['resource'])
        for x in [.1,.2,.3,.5,1.]:instance.append({'instance':i,'method':m,'resource':x,'gain':float(np.interp(x,[r['resource'] for r in rs],[r['gain'] for r in rs]))})
    out=summary(instance,['method','resource'],['gain']);save('B05_marginal_values.csv',out)
    tests=[]
    for x in [.1,.2,.3,.5,1.]:
        rs=[r for r in instance if r['resource']==x]
        for label,b in [('greedy','greedy'),('best_indicator',best_method(rs,'gain'))]:
            tests.append({**compare(rs,'gain',b,resource=x),'selection':label})
    save('B05_marginal_tests.csv',tests)

def B06():
    curves=read(4,'curves_instance.csv')
    bases={(r['instance'],r['kind'],r['method'],r['mode']):r['eta'] for r in curves if r['f']==0}
    selected=[]
    for r in curves:
        if r['mode']!='static' or not any(abs(r['f']-x)<1e-10 for x in [.1,.2,.3,.4]):continue
        v=r.copy();v['eta']=bases[(r['instance'],r['kind'],r['method'],r['mode'])]*r['Phi']/r['Phi_P'] if r['Phi_P']>0 else None
        v['absolute_Phi_gap']=abs(r['Phi_P']-r['Phi']);selected.append(v)
    rows=summary(selected,['kind','method','f'],['Phi','Phi_P','eta','absolute_Phi_gap'])
    for r in rows:r['max_absolute_Phi_gap_instance']=max(v['absolute_Phi_gap'] for v in selected if all(v[k]==r[k] for k in ['kind','method','f']))
    save('B06_PhiP_eta.csv',rows)
    write_json(OUT/'B06_extrema.json',{'max_instance_abs_PhiP_minus_Phi':max(r['absolute_Phi_gap'] for r in selected),'max_mean_abs_PhiP_minus_Phi':max(abs(r['Phi_P']-r['Phi']) for r in rows),'eta_mean_min':min(r['eta'] for r in rows if r['eta'] is not None),'eta_mean_max':max(r['eta'] for r in rows if r['eta'] is not None)})

def B08():
    raw=[r for r in read(6,'radius_accuracy_raw.csv') if r['method']=='SCFC'];out=summary(raw,['radius'],['gain_CMLC','gain_loop','median_teams','F0'])
    for r in out:
        rs=[x for x in raw if x['radius']==r['radius']]
        for key in ['gain_CMLC','gain_loop']:
            v=[x[key] for x in rs];r[key+'_p_two_sided']=float(stats.wilcoxon(v,alternative='two-sided').pvalue) if np.any(v) else 1.
    save('B08_radius.csv',out)
    raw=read(6,'composition_instance.csv');out=summary(raw,['composition','method'],['R_rob'])
    for r in out:
        rs=[x for x in raw if x['composition']==r['composition']];r['SCFC_vs_CMLC_p_two_sided']=compare(rs,'R_rob','CMLC')['p_two_sided']
    save('B08_composition.csv',out)
    raw=read(7,'scale_instance.csv');out=[]
    for n in [30,60,120]:
        rs=[r for r in raw if r['n']==n]
        for metric in ['noise_tau','tau','R_rob','R_F']:
            if metric=='tau':
                source=[r for r in read(2,'metrics_raw.csv') if r['n']==n and r['kind']=='node']
            else:source=rs
            methods=['SCFC'] if metric=='noise_tau' else sorted({r['method'] for r in source})
            for m in methods:
                vals=[r[metric] for r in source if r['method']==m and r[metric] is not None]
                if vals:out.append({'scale':{30:'S1',60:'S2',120:'S3'}[n],'n_nodes':n,'metric':metric,'method':m,**stat(vals,'value')})
            if metric not in ['tau','R_rob','R_F']:continue
            for label,b in [('best_indicator',best_method(source,metric,higher=metric!='R_rob'))]+([('greedy','greedy')] if metric=='R_F' else []):
                out.append({'scale':{30:'S1',60:'S2',120:'S3'}[n],'n_nodes':n,'method':'paired_test','selection':label,**compare(source,metric,b)})
    save('B08_scale_tests.csv',out)
    raw=read(7,'parameter_instance.csv');rows=[]
    for r in raw:
        v=r.copy();v['parameter_value']=next((r[k] for k in ['K','alpha','parameter'] if r.get(k) is not None),None)
        rows.append(v)
    metrics=['tau','node_tau','edge_tau','noise_tau','CMLC_tau','F_cv','rank_stability','mean_team_size','F0','P0','eta0']
    save('B08_parameter_values.csv',summary(rows,['factor','value','parameter_value'],metrics))

def B09():
    rows=[r for r in cohort_objects() if r['kind']=='node'];zero=[r for r in rows if abs(r['CMLC'])<1e-12];out=[]
    per=[]
    for i in range(50):
        z=[r for r in zero if r['instance']==i];per.append({'instance':i,'zero_count':len(z),'truth_zero_share':np.mean([abs((r['truth_loss_1']+r['truth_loss_2'])/2)<1e-12 for r in z]) if z else None})
    out=summary(per,[],['zero_count','truth_zero_share']);out[0].update(zero_CMLC_total=len(zero),truth_zero_among_zero_CMLC=sum(abs((r['truth_loss_1']+r['truth_loss_2'])/2)<1e-12 for r in zero),pooled_truth_zero_share=np.mean([abs((r['truth_loss_1']+r['truth_loss_2'])/2)<1e-12 for r in zero]),truth_rule='mean(truth_loss_1,truth_loss_2); abs < 1e-12')
    save('B09_zero_CMLC.csv',out)
    raw=read(2,'metrics_raw.csv');save('B09_noise_tau.csv',summary([r for r in raw if r['n']==60 and r['method']=='SCFC'],['kind'],['noise_tau']))

def B10():
    out=[]
    def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
    for e in [2,4,5]:
        rs=read(e,'wilcoxon.csv');rs=[r for r in rs if e!=5 or r['kind']=='node'];p=np.array([r['p_two_sided'] for r in rs]);idx=np.argsort(p);adj=np.empty_like(p);adj[idx]=np.minimum(1,np.maximum.accumulate(p[idx]*(len(p)-np.arange(len(p)))))
        for r,a in zip(rs,adj):
            label=';'.join(f'{k}={v}' for k,v in r.items() if k in ['kind','mode','metric','intensity','attack','comparison','baseline'])
            original_star=bool(r.get('star',False))
            out.append({'table':{2:'Table 4',4:'Table 5',5:'Table 6'}[e],'comparison':label,'p_raw':r['p_two_sided'],'p_holm':a,'sig_raw':stars(r['p_two_sided']),'sig_holm':stars(a),'star_changed':stars(r['p_two_sided'])!=stars(a),'binary_significance_changed':(r['p_two_sided']<.05)!=(a<.05),'table_star_raw':'*' if original_star else '', 'table_star_holm':'*' if original_star and a<.05 else '', 'table_star_changed':original_star and a>=.05, 'family_size':len(p)})
    save('B10_holm.csv',out)
    changed=[r for r in out if r['star_changed']]
    if changed:save('B10_changed_cells.csv',changed)
    else:
        with (OUT/'B10_changed_cells.csv').open('w',encoding='utf-8-sig',newline='') as f:
            csv.DictWriter(f,fieldnames=list(out[0])).writeheader()
        print('B10_changed_cells.csv: 0 rows (no changes)',flush=True)

def B12():
    raw=[r for r in read(4,'curves_instance.csv') if r['mode']=='static'];inst=[]
    for kind,m,i in sorted({(r['kind'],r['method'],r['instance']) for r in raw}):
        rs=sorted([r for r in raw if (r['kind'],r['method'],r['instance'])==(kind,m,i)],key=lambda r:r['f'])
        for threshold in [.3,.5,.7]:
            hits=[r['f'] for r in rs if r['Phi']<threshold]
            inst.append({'kind':kind,'method':m,'instance':i,'F_req_fraction':threshold,'f_c':hits[0] if hits else .5,'censored':not bool(hits)})
    save('B12_freq_instance.csv',inst)
    rows=summary(inst,['kind','method','F_req_fraction'],['f_c','censored'])
    for kind in ['node','edge']:
        for th in [.3,.5,.7]:
            rs=sorted([r for r in rows if r['kind']==kind and r['F_req_fraction']==th],key=lambda r:(r['f_c'],r['method']))
            for rank,r in enumerate(rs,1):r['rank_lower_fc']=rank
    save('B12_freq_sensitivity.csv',rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('tasks',nargs='*',default=['A','B01','B05','B06','B08','B09','B10']);args=p.parse_args()
    for task in args.tasks:globals()[task]()

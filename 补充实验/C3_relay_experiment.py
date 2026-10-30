"""C3 core experiment. Run test_relay.py before this module."""
from relay_model import *

def job(i):
    m=RelayModel(i);d,b=construction(m,i);nd,ed,recovery=m.all_scores(b)
    valid=[g for gs in b.task_combinations for g in gs]
    struct={'instance':i,'P0':b.P,'F0':b.F,'eta0':b.E,'communication_edges':len(m.initial_edges),'average_degree':2*len(m.initial_edges)/60,'hyperedges_unique':len(b.hyperedge_counts),'task_labeled_groups':len(valid),'mean_team_size':np.mean([len(g) for g in valid]),'mean_groups_per_task':np.mean([len(gs) for gs in b.task_combinations]),'zero_group_fraction':np.mean([not bool(gs) for gs in b.task_combinations]),'feasible_task_fraction':np.mean(b.feasible_tasks),'position_rejection':np.mean([1-1/t for t in m.position_trials])}
    taskrows=[]
    for k,gs in enumerate(b.task_combinations):
        count=Counter();total=sum(gs.values())
        for g,c in gs.items():
            for node in g:count[node]+=c
        bottleneck=[n for n,c in count.items() if total and c/total>=.9]
        taskrows.append({'instance':i,'task':k,'task_type':m.tasks[k].task_type,'task_name':TASK_NAMES[m.tasks[k].task_type],'minimum_hops':m.tasks[k].minimum_hops,'groups':len(gs),'team_size':np.mean([len(g) for g in gs]) if gs else None,'kappa':max(count.values())/total if total else None,'bottleneck_task':bool(bottleneck) if total else None,**{'bottleneck_'+r:sum(m.roles[n]==r for n in bottleneck) for r in ['S','D','R','I']}})
    accuracy=[];objects=[];loss_by_role=[]
    for kind,scoremap in [('node',nd),('edge',ed)]:
        truth,bases=true_losses(m,i,kind,1000,2);true=truth[:,0];ranking=order(true);noise=correlation(true,truth[:,1]);targets=list(range(60)) if kind=='node' else list(m.initial_edges)
        for method,score in scoremap.items():
            ranks=order(score,method,{'s':nd['s'],'b':m.c_tiebreak})
            accuracy.append({'instance':i,'kind':kind,'method':method,'tau':correlation(score,true),'hit10':len(set(ranks[:10])&set(ranking[:10]))/10,'false5':np.mean(true[ranks[:5]]<.01*bases[0]),'noise_tau':noise})
        for j,obj in enumerate(targets):objects.append({'instance':i,'kind':kind,'object':json.dumps(obj),'role':m.roles[j] if kind=='node' else '', 'truth_loss_1':truth[j,0],'truth_loss_2':truth[j,1],**{method:score[j] for method,score in scoremap.items()}})
        if kind=='node':
            active={n for g in b.hyperedge_counts for n in g}
            for role in ['S','D','R','I']:
                nodes=[j for j in active if m.roles[j]==role]
                for method,vals in [('no_reroute',nd['no_reroute']),('SCFC',nd['SCFC']),('truth',true)]:
                    loss_by_role.append({'instance':i,'role':role,'method':method,'mean_loss':np.mean(vals[nodes]) if nodes else None,'participating_nodes':len(nodes)})
    attacks=[];attack_curves=[]
    for method in ['random','degree','loop','CMLC','SCFC']:
        reps=20 if method=='random' else 1
        for repeat in range(reps):
            ranking=np.random.default_rng(e2.seeds(60,i)[1]+900000000+repeat*997).permutation(60) if method=='random' else order(nd[method])
            states=[NetworkState(set(map(int,ranking[:k])),set()) for k in range(1,31)]
            phi=np.r_[1,e2.batch(m,states,d)/b.F]
            path=[{'f':k/60,'Phi':float(v)} for k,v in enumerate(phi)]
            attacks.append({'instance':i,'method':method,'repeat':repeat,**robustness_half(path)})
            attack_curves.extend({'instance':i,'method':method,'repeat':repeat,**r} for r in path)
    state,events,protocol=damage_path(m,i,'node','SCFC',.2,recovery,d,b)
    rec=[]
    for method in ['random','degree','loop','CMLC','SCFC','greedy']:
        for repeat in range(20 if method=='random' else 1):
            log=recover_full(m,i,state,method,recovery,d,b,repeat,budget=.5)
            metrics,_=recovery_metrics(b,events,protocol,log,.5,120)
            rec.append({'instance':i,'method':method,'repeat':repeat,**metrics})
    return {'structure':struct,'task_rows':taskrows,'accuracy':accuracy,'objects':objects,'role_loss':loss_by_role,'attacks':attacks,'attack_curves':attack_curves,'recovery':rec}

def run():
    # Hard prerequisite is a real passing unit test run, including the no-D shortcut certificate.
    import subprocess
    subprocess.run([sys.executable,'-B',str(HERE/'test_relay.py')],check=True,cwd=HERE)
    data=[]
    for i in range(20):
        data.append(cache_job(f'C3_revised_{i}',job,i));print('C3',i+1,'/20',flush=True)
    structure=[r['structure'] for r in data];tasks=[x for r in data for x in r['task_rows']];acc=[x for r in data for x in r['accuracy']];roles=[x for r in data for x in r['role_loss']];att=[x for r in data for x in r['attacks']];curves=[x for r in data for x in r['attack_curves']];rec=[x for r in data for x in r['recovery']]
    save('C3_structure_instance.csv',structure);save('C3_structure.csv',summary(structure,[],[k for k in structure[0] if k!='instance']))
    save('C3_tasks_instance.csv',tasks)
    taskmetrics=['groups','team_size','kappa','bottleneck_task','bottleneck_S','bottleneck_D','bottleneck_R','bottleneck_I']
    ti=summary(tasks,['instance','task_type','task_name','minimum_hops'],taskmetrics)
    for r in ti:
        selected=[x for x in tasks if x['instance']==r['instance'] and x['task_type']==r['task_type']]
        total=sum(x['bottleneck_'+role] for x in selected for role in ['S','D','R','I'])
        for role in ['S','D','R','I']:
            r['bottleneck_'+role+'_share']=sum(x['bottleneck_'+role] for x in selected)/total if total else None
    taskmetrics+=['bottleneck_'+role+'_share' for role in ['S','D','R','I']]
    save('C3_task_types.csv',summary(ti,['task_type','task_name','minimum_hops'],taskmetrics))
    save('C3_accuracy_instance.csv',acc);accuracy=save('C3_accuracy.csv',summary(acc,['kind','method'],['tau','hit10','false5','noise_tau']))
    save('C3_objects_and_truth.csv',[x for r in data for x in r['objects']])
    save('C3_role_loss_instance.csv',roles);save('C3_role_loss.csv',summary(roles,['role','method'],['mean_loss','participating_nodes']))
    ai=summary(att,['instance','method'],['R_rob','f_c_capped','f_c_censored']);save('C3_attack_instance.csv',ai)
    attacks=save('C3_attack.csv',summary(ai,['method'],['R_rob','f_c_capped','f_c_censored']))
    ac=summary(curves,['instance','method','f'],['Phi']);save('C3_attack_curves.csv',summary(ac,['method','f'],['Phi']))
    ri=summary(rec,['instance','method'],['R_res','R_F','Phi_final']);save('C3_recovery_instance.csv',ri);recovery=save('C3_recovery.csv',summary(ri,['method'],['R_res','R_F','Phi_final']))
    tests=[]
    for kind in ['node','edge']:
        rs=[r for r in acc if r['kind']==kind]
        for metric in ['tau','hit10','false5']:
            best=best_method(rs,metric,exclude=('SCFC','no_reroute','alpha0'),higher=metric!='false5')
            for label,base in [('CMLC','CMLC'),('best_indicator',best),('no_reroute','no_reroute'),('alpha0','alpha0')]:tests.append({'scope':'accuracy','kind':kind,'selection':label,**compare(rs,metric,base)})
    for metric in ['R_rob','f_c_capped']:
        tests.append({'scope':'attack',**compare(ai,metric,'CMLC')})
    for metric in ['R_res','R_F','Phi_final']:
        for base in ['CMLC','greedy']:tests.append({'scope':'recovery',**compare(ri,metric,base)})
    save('C3_tests.csv',tests)
    comparison=[]
    for r in accuracy:
        for metric in ['tau','hit10','false5']:
            comparison.append({'scope':'accuracy','kind':r['kind'],'method':r['method'],'metric':metric,'system':'S2_relay',**{k:r.get(metric+s) for k,s in [('value',''),('value_ci95','_ci95'),('value_n','_n')]}})
    for r in read(2,'accuracy_summary.csv'):
        if r.get('n')!=60:continue
        for metric in ['tau','hit10','false5']:
            comparison.append({'scope':'accuracy','kind':r['kind'],'method':r['method'],'metric':metric,'system':'S2_original','value':r.get(metric),'value_ci95':r.get(metric+'_ci95'),'value_n':r.get(metric+'_n',r.get('instances'))})
    for scope,rows,metrics in [('attack',attacks,['R_rob','f_c_capped']),('recovery',recovery,['R_res','R_F','Phi_final'])]:
        for r in rows:
            for metric in metrics:comparison.append({'scope':scope,'kind':'node','method':r['method'],'metric':metric,'system':'S2_relay','value':r[metric],'value_ci95':r[metric+'_ci95'],'value_n':r[metric+'_n']})
        originals=read(4,'table5_robustness.csv') if scope=='attack' else read(5,'table6_all.csv')
        for r in originals:
            if r['kind']!='node' or (scope=='attack' and r['mode']!='static') or (scope=='recovery' and (r['attack']!='SCFC' or r['intensity']!=.2 or r['budget']!=.5)):continue
            for metric in metrics:comparison.append({'scope':scope,'kind':'node','method':r['method'],'metric':metric,'system':'S2_original','value':r[metric],'value_ci95':r[metric+'_ci95'],'value_n':r[metric+'_n']})
    save('C3_S2_comparison.csv',comparison)
if __name__=='__main__':run()

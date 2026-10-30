from common import *
from itertools import combinations

def B02_job(i,level):
    m=model_for(60,i);d,b=construction(m,i);scores=recovery_scores(m,b)
    state,events,protocol=damage_path(m,i,'mixed','SCFC',level,scores,d,b)
    pairs=set()
    for gs in b.groups:
        for members,g in gs.items():
            if g.efficiency_sum<=0:continue
            for node in state.failed_nodes&members:
                for edge in state.damaged_edges & g.links.keys():pairs.add((node,edge))
    result=[];actions=[]
    for method in ['SCFC','greedy','CMLC','degree']:
        log=recover_full(m,i,state,method,scores,d,b,budget=.5)
        # A fractional prefix of the full action budget is linearly prorated at its boundary.
        quarter=.25*len(log);weights=np.clip(quarter-np.arange(len(log)),0,1)
        node_steps=np.array([a['kind']=='node' for a in log],float)
        prev=events[-1]['F'];gains=[];restored_nodes=set();restored_edges=set()
        for a in log:
            gain=a['F']-prev;prev=a['F'];gains.append(gain)
            (restored_nodes if a['kind']=='node' else restored_edges).add(a['target'])
            actions.append({'instance':i,'kind':'mixed','intensity':level,'method':method,'F0':b.F,'delta_F':gain,**a})
        complete=sum(n in restored_nodes and e in restored_edges for n,e in pairs)
        result.append({'instance':i,'kind':'mixed','intensity':level,'method':method,
            'node_share_first_quarter':float(weights@node_steps/quarter) if quarter else None,
            'node_share_all':float(node_steps.mean()),'mean_delta_F':float(np.mean(gains)),
            'low_gain_share':float(np.mean(np.array(gains)<=.001*b.F)),
            'complement_pair_completion':complete/len(pairs) if pairs else None,
            'complement_pairs':len(pairs),'complement_pairs_restored':complete,
            'Phi_final':log[-1]['F']/b.F})
    return {'rows':result,'actions':actions}

def B02():
    rows=[];actions=[]
    for level in [.2,.3]:
        for i in range(50):
            r=cache_job(f'B02_{level}_{i}',B02_job,i,level);rows+=r['rows'];actions+=r['actions']
            if (i+1)%10==0:print('B02',level,i+1,flush=True)
    save('B02_mixed_mechanism_instance.csv',rows);save('B02_actions.csv',actions)
    metrics=['node_share_first_quarter','node_share_all','mean_delta_F','low_gain_share','complement_pair_completion','complement_pairs','complement_pairs_restored','Phi_final']
    save('B02_mixed_mechanism.csv',summary(rows,['kind','intensity','method'],metrics))
    tests=[]
    for l in [.2,.3]:
        for metric in metrics:
            tests.append(compare([r for r in rows if r['intensity']==l],metric,'greedy',intensity=l))
    save('B02_tests.csv',tests)

def B03_job(i):
    m=model_for(60,i);d,b=construction(m,i);objects=sorted([r for r in cohort_objects() if r['kind']=='node' and r['instance']==i],key=lambda r:r['object']);nodes=[];corr=[]
    for o in objects:
        node=o['object'];role=o['role'];teams=[gs for gs in b.task_combinations if any(node in g for g in gs)]
        row={'instance':i,'node':node,'role':role,'err':(o['truth_loss_1']+o['truth_loss_2'])/2-o['SCFC'],
            'degree':o['degree'],'betweenness':o['betweenness'],'tasks':len(teams),'hyperedges':o['d_H']}
        if teams:
            after=m.evaluate(NetworkState({node},set()),random_inputs=d)
            successful=sum(sum(gs.values()) for gs in after.task_combinations)
            new=sum(count for k,gs in enumerate(after.task_combinations) for g,count in gs.items() if g not in b.task_combinations[k])
            row.update(new_team_share=new/successful if successful else None,new_team_successes=new,successes_after_deletion=successful)
        else:row.update(new_team_share=None,new_team_successes=None,successes_after_deletion=None)
        nodes.append(row)
    for role in ['S','D','I']:
        rs=[r for r in nodes if r['role']==role]
        for feature in ['degree','betweenness','tasks','hyperedges']:
            x=np.array([r[feature] for r in rs]);y=np.array([r['err'] for r in rs]);sp=stats.spearmanr(x,y) if np.ptp(x)>0 and np.ptp(y)>0 else None
            corr.append({'instance':i,'role':role,'feature':feature,'rho':float(sp.statistic) if sp else None,'spearman_p_two_sided':float(sp.pvalue) if sp else None})
    return {'nodes':nodes,'correlations':corr}

def B03():
    nodes=[];corr=[]
    for i in range(50):
        r=cache_job(f'B03_{i}',B03_job,i);nodes+=r['nodes'];corr+=r['correlations']
        if (i+1)%10==0:print('B03',i+1,flush=True)
    save('B03_nodes.csv',nodes);save('B03_correlations_instance.csv',corr)
    inst=summary(nodes,['instance','role'],['err','new_team_share'])
    out=summary(inst,['role'],['err','new_team_share'])
    for r in out:r['section']='role_mean';r['feature']=''
    co=summary(corr,['role','feature'],['rho'])
    for r in co:
        v=[x['rho'] for x in corr if x['role']==r['role'] and x['feature']==r['feature'] and x['rho'] is not None]
        r.update(section='within_scene_spearman',p_two_sided=float(stats.wilcoxon(v,alternative='two-sided').pvalue) if np.any(v) else 1.)
    save('B03_D_bias.csv',out+co)
    tests=[]
    for other in ['S','I']:
        a={r['instance']:r['new_team_share'] for r in inst if r['role']=='D'};b={r['instance']:r['new_team_share'] for r in inst if r['role']==other}
        tests.append({'comparison':'D - '+other,**test(a,b)})
    save('B03_role_tests.csv',tests)

def B04():
    objs=cohort_objects();metrics=read(2,'metrics_raw.csv');out=[]
    for i in range(50):
        m=model_for(60,i);_,b=construction(m,i);rs=sorted([r for r in objs if r['instance']==i and r['kind']=='node'],key=lambda r:r['object'])
        for method in ['SCFC','s','CMLC','c']:
            ids=order([r[method] for r in rs],method,{k:[r[k] for r in rs] for k in ['s','b']})[:5];chosen=[rs[j]['object'] for j in ids]
            e=next(r for r in metrics if r['n']==60 and r['instance']==i and r['kind']=='node' and r['method']==method)
            individual=sum((rs[j]['truth_loss_1']+rs[j]['truth_loss_2'])/2 for j in ids);joint=e['F0']*(1-e['Phi5'])
            share=np.mean([any(a in g and c in g for gs in b.task_combinations for g in gs) for a,c in combinations(chosen,2)])
            out.append({'instance':i,'method':method,'joint':joint,'sum':individual,'ratio':joint/individual if abs(individual)>1e-12 else None,'co_task_share':float(share),'top5':json.dumps(chosen)})
    save('B04_top5_instance.csv',out);save('B04_top5_nonadditivity.csv',summary(out,['method'],['joint','sum','ratio','co_task_share']))

def B07_job(i):
    m=model_for(60,i);d,b=construction(m,i);scores=recovery_scores(m,b)
    objects={r['object']:r for r in cohort_objects() if r['instance']==i and r['kind']=='node'};out=[]
    for attack in ['random','degree','SCFC']:
        for repeat in range(20):
            state,events,prot=damage_path(m,i,'node',attack,.2,scores,d,b,repeat if attack=='random' else 0)
            for method in RECOVERY:
                # Deterministic attack/recovery combinations are evaluated once; repeats are exact duplicates.
                if repeat and attack!='random' and method!='random':continue
                log=recover_full(m,i,state,method,scores,d,b,repeat,budget=.5)
                restored={a['target'] for a in log if a['kind']=='node'};remaining=state.failed_nodes-restored
                phi=log[-1]['F']/b.F
                for role in ['all','S','D','I']:
                    ns=[n for n in remaining if role=='all' or m.roles[n]==role]
                    out.append({'instance':i,'attack':attack,'method':method,'repeat':repeat,'role':role,'remaining_count':len(ns),'role_share':len(ns)/len(remaining),
                        'negative_truth_share':np.mean([(objects[n]['truth_loss_1']+objects[n]['truth_loss_2'])/2<0 for n in ns]) if ns else None,
                        'negative_SCFC_share':np.mean([objects[n]['SCFC']<0 for n in ns]) if ns else None,
                        'Phi_final_gt1':phi>1,'Phi_final':phi,'effective_repeats':20 if attack!='random' and method!='random' else 1})
    return out

def B07():
    rows=[]
    for i in range(50):
        rows+=cache_job(f'B07_{i}',B07_job,i)
        if (i+1)%5==0:print('B07',i+1,flush=True)
    save('B07_trap_runs.csv',rows)
    metrics=['remaining_count','role_share','negative_truth_share','negative_SCFC_share','Phi_final_gt1','Phi_final']
    inst=summary(rows,['instance','attack','method','role'],metrics)
    save('B07_trap_nodes_instance.csv',inst)
    save('B07_trap_nodes.csv',summary(inst,['attack','method','role'],metrics))

def B11():
    rows=[]
    for i in range(50):
        m=model_for(60,i);_,b=construction(m,i)
        for k,gs in enumerate(b.task_combinations):
            total=sum(gs.values());counts=Counter()
            for g,v in gs.items():
                for n in g:counts[n]+=v
            kap=max(counts.values())/total if total else None
            for th in [.8,.9,.95,1.]:rows.append({'instance':i,'task':k,'task_type':m.tasks[k].task_type,'threshold':th,'kappa':kap,'bottleneck_share':float(kap>=th) if kap is not None else None})
    save('B11_task_kappa.csv',rows)
    inst=summary(rows,['instance','task_type','threshold'],['bottleneck_share'])
    save('B11_bottleneck_threshold.csv',summary(inst,['task_type','threshold'],['bottleneck_share']))

def B13():
    rows=[]
    for hops in [8,10,12,16,20]:
        for i in range(10):
            m=model_for(60,i,max_hops=hops);_,b=construction(m,i);overlaps=[];sizes=[]
            for gs in b.task_combinations:
                members=[set(j for j in g if m.roles[j]!='I') for g in gs];sizes.extend(len(g) for g in gs)
                overlaps.extend(len(g&h) for g,h in combinations(members,2))
            rows.append({'instance':i,'L_max':hops,'overlap_mean':np.mean(overlaps) if overlaps else None,'overlap_ge3_share':np.mean(np.array(overlaps)>=3) if overlaps else None,'mean_team_size':np.mean(sizes),'teams_per_task':np.mean([len(gs) for gs in b.task_combinations])})
        print('B13',hops,flush=True)
    save('B13_lmax_overlap_instance.csv',rows);save('B13_lmax_overlap.csv',sorted(summary(rows,['L_max'],['overlap_mean','overlap_ge3_share','mean_team_size','teams_per_task']),key=lambda r:r['L_max']))

def B14():
    objs=cohort_objects();rows=[]
    for i in range(10):
        m=model_for(60,i);_,b=construction(m,i);E=len(m.initial_edges)
        truth=np.array([r['truth_loss_1'] for r in sorted([r for r in objs if r['instance']==i and r['kind']=='edge'],key=lambda r:m.edge_index[tuple(json.loads(r['object']))])])
        for alpha in [0,1,2,3,5,8]:
            numerator=denominator=success_num=success_den=0.
            for record in b.groups:
                entries=list(record.items());G=len(entries)
                if not G:continue
                count=np.array([v.count for g,v in entries],float);remove=np.zeros((G,E));members=np.zeros((G,m.config.node_count));successful=np.array([v.efficiency_sum>0 for g,v in entries])
                for j,(g,v) in enumerate(entries):
                    members[j,[n for n in g if m.roles[n]!='I']]=1
                    for edge,(cnt,eta) in v.links.items():remove[j,m.edge_index[edge]]=cnt
                used=np.flatnonzero(remove.sum(axis=0))
                if not len(used):continue
                remove=remove[:,used];alive=count[:,None]-remove;overlap=(members@members.T)**alpha;den=overlap@alive
                fallback=np.divide(alive,alive.sum(axis=0),out=np.zeros_like(alive),where=alive.sum(axis=0)>0)
                share=np.divide(np.diag(overlap)[:,None]*alive,den,out=fallback.copy(),where=den>0)
                numerator+=float(np.sum(remove*share));denominator+=float(remove.sum())
                success_num+=float(np.sum((remove*share)[successful]));success_den+=float(remove[successful].sum())
            _,sc=m.scfc(b,alpha=alpha,kind='edge');score=np.array([sc[e] for e in m.initial_edges])
            rows.append({'instance':i,'alpha':alpha,'self_substitution_share':numerator/denominator if denominator else None,'effective_group_self_share':success_num/success_den if success_den else None,'destroyed_occurrences':denominator,'edge_tau':correlation(score,truth)})
        print('B14',i+1,flush=True)
    save('B14_link_self_substitution_instance.csv',rows);save('B14_link_self_substitution.csv',summary(rows,['alpha'],['self_substitution_share','effective_group_self_share','destroyed_occurrences','edge_tau']))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('tasks',nargs='+');a=p.parse_args()
    for task in a.tasks:globals()[task]()

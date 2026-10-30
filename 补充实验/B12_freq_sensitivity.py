"""Recompute exact removal-step crossings; repeat averages are taken after censoring."""
from common import *

def job(i):
    m=model_for(60,i);d,b=construction(m,i);objs=cohort_objects();rows=[]
    for kind,methods in [('node',ATTACK_NODE),('edge',ATTACK_EDGE)]:
        targets=list(range(60)) if kind=='node' else list(m.initial_edges)
        rs=[r for r in objs if r['instance']==i and r['kind']==kind]
        rs=sorted(rs,key=lambda r:r['object'] if kind=='node' else m.edge_index[tuple(json.loads(r['object']))])
        total=len(targets);steps=math.ceil(total*.5);f=np.arange(steps+1)/total
        for method in methods:
            for repeat in range(20 if method=='random' else 1):
                seed=e2.seeds(60,i)[1]+900000000+repeat*997+(kind=='edge')*100000
                rank=np.random.default_rng(seed).permutation(total) if method=='random' else order([r[method] for r in rs])
                states=[]
                for k in range(1,steps+1):
                    chosen={targets[j] for j in rank[:k]}
                    states.append(NetworkState(chosen,set()) if kind=='node' else NetworkState(set(),chosen))
                phi=np.r_[1.,e2.batch(m,states,d)/b.F]
                for th in [.3,.5,.7]:
                    hits=np.flatnonzero((phi<th)&(f<=.5+1e-12))
                    rows.append({'instance':i,'kind':kind,'method':method,'repeat':repeat,'F_req_fraction':th,'f_c':float(f[hits[0]]) if len(hits) else .5,'censored':not bool(len(hits))})
    return rows

def run():
    runs=[]
    for i in range(50):
        runs+=cache_job(f'B12_exact_{i}',job,i)
        if (i+1)%5==0:print('B12',i+1,flush=True)
    inst=summary(runs,['instance','kind','method','F_req_fraction'],['f_c','censored'])
    save('B12_freq_runs.csv',runs);save('B12_freq_instance.csv',inst)
    rows=summary(inst,['kind','method','F_req_fraction'],['f_c','censored'])
    for kind in ['node','edge']:
        for th in [.3,.5,.7]:
            rs=sorted([r for r in rows if r['kind']==kind and r['F_req_fraction']==th],key=lambda r:(r['f_c'],r['method']))
            for rank,r in enumerate(rs,1):r['rank_lower_fc']=rank
    save('B12_freq_sensitivity.csv',rows)
if __name__=='__main__':run()

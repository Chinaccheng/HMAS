from common import *

def job(i):
    m=model_for(60,i);d,b=construction(m,i);rows=[];met=[]
    for role in ['S','D','I','all_random_reference']:
        ns=np.arange(60) if role=='all_random_reference' else np.flatnonzero(m.roles==role)
        for repeat in range(20):
            seed=e2.seeds(60,i)[1]+900000000+repeat*997
            chosen=np.random.default_rng(seed).permutation(ns)[:12]
            states=[NetworkState(set(map(int,chosen[:k])),set()) for k in range(1,13)]
            phi=np.r_[1.,e2.batch(m,states,d)/b.F];f=np.arange(13)/60
            for k,p in enumerate(phi):rows.append({'instance':i,'role':role,'repeat':repeat,'f_V':k/60,'Phi':float(p)})
            hits=np.flatnonzero(phi<.5)
            met.append({'instance':i,'role':role,'repeat':repeat,'R_rob':float(np.trapezoid(phi,f)/.2),'f_c':float(f[hits[0]]) if len(hits) else .2,'censored':not bool(len(hits))})
    return {'curves':rows,'metrics':met}

def run():
    curves=[];metrics=[]
    for i in range(50):
        r=cache_job(f'C2_with_reference_{i}',job,i);curves+=r['curves'];metrics+=r['metrics']
        if (i+1)%5==0:print('C2',i+1,flush=True)
    ci=summary(curves,['instance','role','f_V'],['Phi']);mi=summary(metrics,['instance','role'],['R_rob','f_c','censored'])
    save('C2_role_attack_instance.csv',mi);save('C2_role_attack_curves.csv',summary(ci,['role','f_V'],['Phi']))
    out=summary(mi,['role'],['R_rob','f_c','censored'])
    save('C2_role_attack_summary.csv',out)
    tests=[]
    for a,b in [('S','D'),('S','I'),('D','I')]:
        for metric in ['R_rob','f_c']:
            aa={r['instance']:r[metric] for r in mi if r['role']==a};bb={r['instance']:r[metric] for r in mi if r['role']==b}
            tests.append({'comparison':a+' - '+b,'metric':metric,**test(aa,bb)})
    save('C2_role_attack_tests.csv',tests)
if __name__=='__main__':run()

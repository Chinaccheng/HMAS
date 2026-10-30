"""Serial, warmed, wall-clock timing. Run after B10 and before P2."""
from common import *

def timed(rows,n,i,item,func,**extra):
    t=time.perf_counter();value=func();elapsed=time.perf_counter()-t
    rows.append({'scale':{30:'S1',60:'S2',120:'S3'}[n],'instance':i,'item':item,'seconds':elapsed,**extra})
    return value

def run():
    # Warm every JIT search path and task-loop kernel before any reported timer.
    warm=model_for(30,0);d,b=construction(warm,0);warm.scfc(b);state_loops(warm)
    e2.batch(warm,[NetworkState({0},set())],draws_for(warm,0,1000))
    rows=[];calls=[]
    for n in [30,60,120]:
        for i in range(5):
            m=model_for(n,i);d=draws_for(m,i,300)
            b=timed(rows,n,i,'construct_H0',lambda:m.evaluate(random_inputs=d))
            timed(rows,n,i,'SCFC_node',lambda:m.scfc(b,kind='node'))
            timed(rows,n,i,'SCFC_edge',lambda:m.scfc(b,kind='edge'))
            timed(rows,n,i,'CMLC_node_and_edge',lambda:e2.cmlc(m,b))
            timed(rows,n,i,'loops_node_and_edge',lambda:state_loops(m))
            g=m.graph()
            timed(rows,n,i,'degree',lambda:np.array(list(dict(g.degree()).values())))
            timed(rows,n,i,'betweenness',lambda:nx.betweenness_centrality(g))
            timed(rows,n,i,'edge_betweenness',lambda:nx.edge_betweenness_centrality(g))
            truth=draws_for(m,i,1000)
            # The intact truth baseline is timed separately and included in each deletion total.
            t=time.perf_counter();m.evaluate(random_inputs=truth,collect=False);bt=time.perf_counter()-t
            for kind,targets in [('node',list(range(n))),('edge',list(m.initial_edges))]:
                def calculate():
                    for obj in targets:
                        state=NetworkState({obj},set()) if kind=='node' else NetworkState(set(),{obj})
                        m.evaluate(state,random_inputs=truth,collect=False)
                timed(rows,n,i,'truth_'+kind,calculate,objects_measured=len(targets),objects_total=len(targets),extrapolated=False)
                rows[-1]['seconds']+=bt
            scores=recovery_scores(m,b);state,events,protocol=damage_path(m,i,'node','SCFC',.2,scores,d,b)
            for method in ['SCFC','greedy']:
                log=timed(rows,n,i,'recovery_'+method,lambda:recover_full(m,i,state,method,scores,d,b))
                calls.append({'scale':{30:'S1',60:'S2',120:'S3'}[n],'instance':i,'method':method,'candidate_evaluations':sum(a['candidate_calls'] for a in log),'recovery_steps':len(log)})
            save('C1_timing.csv',rows);save('C1_candidate_counts.csv',calls)
            print('C1 completed',n,i,flush=True)
    save('C1_timing_summary.csv',summary(rows,['scale','item'],['seconds']))
    save('C1_candidate_counts_summary.csv',summary(calls,['scale','method'],['candidate_evaluations','recovery_steps']))
if __name__=='__main__':run()

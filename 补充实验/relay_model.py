from common import *
from relay_definition import *
from relay_kernel import UAVModel as GeneralizedModel,Task
from relay_loops import task_loops
from dataclasses import dataclass

@dataclass(frozen=True)
class RelayConfig(Config):
    n_s:int=20
    n_d:int=10
    n_r:int=10
    n_i:int=20
    @property
    def node_count(self):return self.n_s+self.n_d+self.n_r+self.n_i
    def validate(self):
        super().validate()
        if self.n_r<=0:raise ValueError('n_r must be positive')
        return self

class RelayModel(GeneralizedModel):
    def __init__(self,index):
        self.config=RelayConfig().validate();self.scene_seed=scenario.seeds(60,index)[0]
        cfg=self.config;rng=np.random.default_rng(self.scene_seed)
        self.positions=rng.uniform(0,cfg.region_size,(cfg.node_count,2))
        self.roles=np.array(['S']*cfg.n_s+['D']*cfg.n_d+['R']*cfg.n_r+['I']*cfg.n_i);rng.shuffle(self.roles)
        dist=np.linalg.norm(self.positions[:,None]-self.positions[None,:],axis=2)
        self.initial_edges=tuple((int(a),int(b)) for a,b in zip(*np.where(np.triu(dist<=cfg.communication_radius,1))))
        self.edge_set=set(self.initial_edges);self.edge_index={e:j for j,e in enumerate(self.initial_edges)}
        task_pos=rng.uniform(0,cfg.region_size,(cfg.task_count,2));types=np.repeat(np.arange(5),cfg.task_count//5);rng.shuffle(types)
        self.tasks=[Task(k,tuple(map(float,p)),int(t),REQUIREMENTS[t]) for k,(p,t) in enumerate(zip(task_pos,types))]
        self.coverage_s=[set() for t in self.tasks];self.coverage_i=[set() for t in self.tasks]
        for k,t in enumerate(self.tasks):scenario.set_task_position(self,k,t.position)
        # Match the original seed and rejection sequence; never regenerate the network.
        rng=np.random.default_rng(self.scene_seed+900000000);self.position_trials=[]
        for k in range(len(self.tasks)):
            for attempt in range(1000000):
                if attempt:scenario.set_task_position(self,k,rng.uniform(0,cfg.region_size,2))
                if self.reachable_execution_nodes(k):break
                if attempt==10000 and not scenario._has_feasible_position(self,k):raise scenario.InfeasibleScenario(f'C3 index={index}, task={k}: no feasible location')
            else:raise RuntimeError('one million position proposals exhausted')
            self.position_trials.append(attempt+1)

    def all_scores(self,base):
        s=self.scores(base);a,b=self.scfc(base);ln,le,_=task_loops(self);cn,ce=e2.cmlc(self,base)
        self.c_tiebreak=s.b
        nd={'degree':s.degree,'betweenness':s.betweenness,'loop':ln,'s':s.s,'c':s.c,'CMLC':cn,'SCFC':a}
        ed={'degree':np.array([self.graph().degree(a)+self.graph().degree(b) for a,b in self.initial_edges]),'betweenness':np.array([s.edge_betweenness[e] for e in self.initial_edges]),'loop':le,'u':np.array([s.u[e] for e in self.initial_edges]),'h':np.array([s.h[e] for e in self.initial_edges]),'CMLC':ce,'SCFC':np.array([b[e] for e in self.initial_edges])}
        for method,kw in [('no_reroute',{'reroute':False}),('alpha0',{'alpha':0})]:
            a,b=self.scfc(base,**kw);nd[method]=a;ed[method]=np.array([b[e] for e in self.initial_edges])
        recovery={k:(nd[k].copy(),ed[k].copy()) for k in ['degree','betweenness','loop','CMLC','SCFC']}
        for method in ['degree','betweenness']:
            aa,bb=recovery[method];recovery[method]=(aa/max(np.max(abs(aa)),1e-12),bb/max(np.max(abs(bb)),1e-12))
        return nd,ed,recovery

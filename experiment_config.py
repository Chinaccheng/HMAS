"""v11统一场景入口：固定基准配置，仅对任务位置做可行性拒绝采样。"""
from dataclasses import replace
import hashlib,json
import numpy as np
from .config import Config
from .model import UAVModel

SEED=20261026
BASE_K=300
TRUTH_K=1000
GAIN_K=300


class InfeasibleScenario(ValueError):
    """固定网络/任务要求下无法生成可行位置；不得换网络或伪造零值。"""


def config(n=60,K=BASE_K,**changes):
    base=Config(n_s=2*n//5,n_d=n//5,n_i=2*n//5,
        region_size=100*np.sqrt(n/60),communication_radius=25,sensing_radius=30,
        execution_radius=20,task_count={30:25,60:50,120:100}[n],
        priority_weight=10,max_hops=12,search_count=K,validation_search_count=1000,
        attack_time=5,recovery_start=14,horizon={30:120,60:120,120:120}[n],recovery_target=.9)
    return replace(base,**changes)


def seeds(n,index,pilot=False):
    offset=2_000_000_000 if pilot else 0
    return SEED+offset+n*100003+index*1009,SEED+offset+n*100019+index*1013


def set_task_position(model,k,position):
    position=tuple(map(float,position))
    model.tasks[k]=replace(model.tasks[k],position=position)
    distance=np.linalg.norm(model.positions-np.asarray(position),axis=1)
    model.coverage_s[k]=set(map(int,np.flatnonzero((model.roles=='S')&(distance<=model.config.sensing_radius))))
    model.coverage_i[k]=set(map(int,np.flatnonzero((model.roles=='I')&(distance<=model.config.execution_radius))))


def _has_feasible_position(model,k):
    """覆盖条件单调；检查圆盘与区域边界的全部顶点及圆心，排除所有位置都无解的情形。"""
    L=model.config.region_size;pos=model.positions
    S=np.flatnonzero(model.roles=='S');I=np.flatnonzero(model.roles=='I')
    ids=list(S)+list(I);centers=pos[ids];radii=np.array([model.config.sensing_radius]*len(S)+[model.config.execution_radius]*len(I))
    points=[*centers, np.array([0.,0.]),np.array([0.,L]),np.array([L,0.]),np.array([L,L])]
    for c,r in zip(centers,radii):
        for e in (0.,L):
            for d in (0,1):
                h=r*r-(e-c[d])**2
                if h>=-1e-8:
                    for sign in (-1,1):
                        q=c.copy();q[d]=e;q[1-d]=c[1-d]+sign*np.sqrt(max(0,h));points.append(q)
    for i,(c,r) in enumerate(zip(centers,radii)):
        for j in range(i):
            other=centers[j];rr=radii[j];v=other-c;d=np.linalg.norm(v)
            if d<1e-12 or d>r+rr+1e-8 or d<abs(r-rr)-1e-8:continue
            a=(r*r-rr*rr+d*d)/(2*d);h2=r*r-a*a
            if h2 < -1e-8:continue
            mid=c+a*v/d;w=np.array([-v[1],v[0]])/d*np.sqrt(max(0,h2))
            points.extend([mid+w,mid-w])
    oldS,oldI=model.coverage_s[k],model.coverage_i[k];seen=set()
    try:
        for q in points:
            if np.any(q < -1e-8) or np.any(q > L+1e-8):continue
            q=np.clip(q,0,L);dist=np.linalg.norm(pos-q,axis=1)
            cs=frozenset(map(int,S[dist[S]<=model.config.sensing_radius+1e-7]));ci=frozenset(map(int,I[dist[I]<=model.config.execution_radius+1e-7]))
            if not cs or not ci or (cs,ci) in seen:continue
            seen.add((cs,ci));model.coverage_s[k],model.coverage_i[k]=set(cs),set(ci)
            if model.reachable_execution_nodes(k):return True
        return False
    finally:model.coverage_s[k],model.coverage_i[k]=oldS,oldI


def screened_model(n,index,pilot=False,K=BASE_K,task_composition='uniform',**changes):
    scene_seed,_=seeds(n,index,pilot)
    model=UAVModel(config(n,K,**changes),scene_seed)
    if task_composition!='uniform':
        if task_composition not in ('basic','composite'):raise ValueError(task_composition)
        requirement_rng=np.random.default_rng(scene_seed+1771)
        model.tasks=[replace(t,task_type=0 if task_composition=='basic' else 4,
            requirements=0 if task_composition=='basic' else t.requirements if t.task_type==4
                else int(sum(requirement_rng.choice([1,2,4],2,replace=False)))) for t in model.tasks]
    # In extreme role/radius scans a task can be impossible at every position.
    # All-S/all-I coverage gives an upper bound on feasibility; failure certifies
    # impossibility, without discarding or regenerating the network.
    if changes or task_composition!='uniform':
        checked=set()
        for k,t in enumerate(model.tasks):
            if t.requirements in checked:continue
            checked.add(t.requirements);cs,ci=model.coverage_s[k],model.coverage_i[k]
            model.coverage_s[k]=set(map(int,np.flatnonzero(model.roles=='S')))
            model.coverage_i[k]=set(map(int,np.flatnonzero(model.roles=='I')))
            possible=bool(model.reachable_execution_nodes(k))
            model.coverage_s[k],model.coverage_i[k]=cs,ci
            if not possible:raise InfeasibleScenario(f'N={n}, instance={index}, requirements={t.requirements}: no legal path at any task position')
    rng=np.random.default_rng(scene_seed+900_000_000)
    generation=[]
    for k,task in enumerate(model.tasks):
        trials=[]
        for attempt in range(1_000_000):
            if attempt==10000 and not _has_feasible_position(model,k):
                raise InfeasibleScenario(f'N={n}, instance={index}, task={k}: circle-arrangement certificate; no feasible task position')
            if attempt:set_task_position(model,k,rng.uniform(0,model.config.region_size,2))
            feasible=bool(model.reachable_execution_nodes(k))
            trials.append({'position':model.tasks[k].position,'feasible':feasible})
            if feasible:break
        else:raise RuntimeError(f'N={n}, instance={index}, task={k}: feasible position exists but one million uniform proposals failed; stop rather than censor a feasible case')
        generation.append({'task':k,'type':task.task_type,'requirements':task.requirements,
            'attempted':len(trials),'rejected':len(trials)-1,'trials':trials})
    return model,generation


def truth_inputs(model,index,repeat=0,K=TRUTH_K):
    if repeat==0:return search_inputs(model,index,K=K)
    seed=seeds(model.config.node_count,index)[1]+repeat*700_000_000
    return np.random.default_rng(seed).random((len(model.tasks),max(1000,K),21))[:,:K,:model.config.max_hops+1]


def manifest(model,index,ev=None):
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    row={'n':model.config.node_count,'instance':index,'scene_seed':model.scene_seed,
        'search_seed':seeds(model.config.node_count,index)[1],
        'network_sha256':digest([model.positions.tolist(),model.roles.tolist(),model.initial_edges]),
        'tasks_sha256':digest([(t.task_id,t.position,t.task_type,t.requirements) for t in model.tasks]),
        **{k:getattr(model.config,k) for k in ['n_s','n_d','n_i','task_count','priority_weight','max_hops',
            'sensing_radius','communication_radius','execution_radius']}}
    if ev is not None:row.update(K=ev.search_count,F0=ev.F,P0=ev.P,successful_tasks=int(np.count_nonzero(ev.f_task>0)))
    return row


def search_inputs(model,index,pilot=False,K=300):
    seed=seeds(model.config.node_count,index,pilot)[1]
    # 固定每任务1000次、21个跳数随机数，再取前缀，方便后续跨参数配对。
    return np.random.default_rng(seed).random((len(model.tasks),max(1000,K),21))[:,:K,:model.config.max_hops+1]

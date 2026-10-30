import numpy as np
from numba import njit
from relay_definition import *
@njit(cache=False)
def _simple_paths(neighbors,degrees,roles,starts,ends,requirement,limit,edge_ids):
    """完整枚举长度有界的简单路径；可达性动态规划只作安全剪枝。"""
    n=len(roles);E=int(edge_ids.max())+1
    node=np.zeros(n,np.int64);edge=np.zeros(E,np.int64);total=0
    possible=np.zeros((limit+1,n,STATE_COUNT),np.bool_)
    for i in range(n):
        if ends[i]:possible[0,i,0]=True
    for d in range(1,limit+1):
        possible[d]=possible[d-1]
        for i in range(n):
            if roles[i]==TERMINAL_CODE:continue
            for rem in range(STATE_COUNT):
                for q in range(degrees[i]):
                    j=neighbors[i,q];a,b=roles[i],roles[j]
                    if not ALLOW_MATRIX[a,b]:continue
                    if not (starts[j] or UNRESTRICTED_MASK[b] or (rem==0 and ends[j])):continue
                    bit=BIT_MATRIX[a,b]
                    if possible[d-1,j,rem & ~bit]:possible[d,i,rem]=True;break
    path=np.empty(limit+1,np.int32);remaining=np.empty(limit+1,np.int32)
    cursor=np.zeros(limit+1,np.int32);visited=np.zeros(n,np.bool_)
    for start in range(n):
        if not starts[start] or not possible[limit,start,requirement]:continue
        path[0]=start;remaining[0]=requirement;cursor[0]=0;visited[start]=True;depth=0
        while depth>=0:
            i=path[depth];rem=remaining[depth]
            if cursor[depth]>=degrees[i] or depth==limit:
                visited[i]=False;depth-=1;continue
            j=neighbors[i,cursor[depth]];cursor[depth]+=1
            if visited[j]:continue
            a,b=roles[i],roles[j]
            if not ALLOW_MATRIX[a,b]:continue
            if not (starts[j] or UNRESTRICTED_MASK[b] or (rem==0 and ends[j])):continue
            bit=BIT_MATRIX[a,b]
            rest=rem & ~bit
            if b==TERMINAL_CODE:
                if rest==0 and ends[j]:
                    total+=1;node[j]+=1;edge[edge_ids[i,j]]+=1
                    for h in range(depth+1):node[path[h]]+=1
                    for h in range(depth):edge[edge_ids[path[h],path[h+1]]]+=1
                continue
            if depth+1>=limit or not possible[limit-depth-1,j,rest]:continue
            depth+=1;path[depth]=j;remaining[depth]=rest;cursor[depth]=0;visited[j]=True
    return node,edge,total

def task_loops(model):
    n=model.config.node_count;g=model.graph();width=max(dict(g.degree()).values())
    neighbors=np.full((n,width),-1,np.int32);degrees=np.zeros(n,np.int32)
    for i in g:
        row=sorted(g[i]);neighbors[i,:len(row)]=row;degrees[i]=len(row)
    ids=np.full((n,n),-1,np.int32)
    for j,(a,b) in enumerate(model.initial_edges):ids[a,b]=ids[b,a]=j
    roles=np.array([ROLE_CODES[r] for r in model.roles])
    nodes=np.zeros(n,np.int64);edges=np.zeros(len(model.initial_edges),np.int64);totals=[]
    for k,t in enumerate(model.tasks):
        starts=np.zeros(n,np.bool_);ends=np.zeros(n,np.bool_)
        starts[list(model.coverage_s[k])]=True;ends[list(model.coverage_i[k])]=True
        a,b,count=_simple_paths(neighbors,degrees,roles,starts,ends,t.requirements,t.minimum_hops+2,ids)
        nodes+=a;edges+=b;totals.append(count)
    return nodes,edges,totals

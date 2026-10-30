"""v11统一基准实验一：任务可行性筛选、K300/lambda10/12跳、图4与图5三联图、团队份额与瓶颈成员统计。"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_CACHE_DIR","/private/tmp/uav-numba-cache")
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/uav-mpl-v7')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import csv
import json
import math
import shutil
import sys
sys.dont_write_bytecode=True
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats,ndimage

ROOT=Path(__file__).resolve().parent
UAV=ROOT.parent
OUT=ROOT/'result'
sys.path.insert(0,str(UAV.parent))
from UAV.experiment_config import SEED,config,seeds,screened_model,search_inputs,set_task_position
from UAV.model import UAVModel
TASKS=['Basic','S–S','D–D','Feedback','Composite']
COLORS=['#3877AD','#47A6A0','#DC9C39','#A86EAA','#CA5B58']
ROLE_COLORS={'S':'#3877AD','D':'#DE8A36','I':'#428A61'}
MARKERS={'S':'o','D':'s','I':'^'}
TEAM_COLORS=['#8F5AA7','#169C9B']
BOTTLENECK_COLUMN='存在瓶颈成员的任务比例（某成员覆盖 ≥90% 成功搜索）'


def write_json(path,obj):
    def clean(x):
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        return x
    Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False))


def write_csv(name,rows):
    if not rows:return
    if name=='table3_task_types.csv':
        names={'bottleneck_task_fraction':BOTTLENECK_COLUMN,
            'bottleneck_task_fraction_ci95':'瓶颈任务比例的95%CI半宽',
            'bottleneck_task_fraction_n':'瓶颈任务比例的独立场景数'}
        rows=[{names.get(k,k):v for k,v in r.items()} for r in rows]
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def mean_ci(values):
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(a):return None,None,0
    return float(a.mean()),float(stats.t.ppf(.975,len(a)-1)*stats.sem(a)) if len(a)>1 else None,len(a)


def med(values):
    a=[x for x in values if x is not None and np.isfinite(x)]
    return float(np.median(a)) if a else None


def worker(index,pilot=False):
    model,generation=screened_model(60,index,pilot)
    ev=model.evaluate(random_inputs=search_inputs(model,index,pilot))
    assert ev.feasible_tasks.all()
    sd=set(np.flatnonzero(model.roles!='I'));records=[]
    for k,t in enumerate(model.tasks):
        counter=ev.task_combinations[k];groups=sorted(counter,key=lambda g:(-counter[g],tuple(sorted(g))))
        total=sum(counter.values());J=sum((c/total)**2 for c in counter.values()) if total else None
        overlap=Counter(min(len((a&b)&sd),3) for a,b in combinations(groups,2))
        member_counts=Counter()
        for group in groups:
            for member in group:member_counts[member]+=int(counter[group])
        maximum=max(member_counts.values(),default=0)
        bottlenecks=[int(i) for i,c in member_counts.items() if 10*c>=9*total] if total else []
        assert sum(counter.values())<=300
        assert total==0 or max(member_counts.values())<=total
        top_share=int(counter[groups[0]])/total if total else None
        records.append({'task':k,'type':t.task_type,'position':t.position,'requirements':t.requirements,
            'minimum_hops':t.minimum_hops,'effective_groups':len(groups),'successful_searches':total,
            'J':J,'N_eff':1/J if J else None,'occurrences_per_team':total/len(groups) if groups else None,
            'P':float(ev.p_task[k]),'F':float(ev.f_task[k]),'feasible':bool(ev.feasible_tasks[k]),
            'top1_share':top_share,'max_member_share':maximum/total if total else None,
            'bottleneck_member_count':len(bottlenecks),'has_bottleneck':bool(bottlenecks),
            **{f'bottleneck_{role}_count':sum(model.roles[i]==role for i in bottlenecks) for role in 'SDI'},
            'has_unavoidable_member':bool(total and maximum==total),
            'sizes':[len(g) for g in groups],'overlap_hist':[overlap.get(i,0) for i in range(4)],
            'groups':[{'members':sorted(g),'count':int(counter[g]),'share':counter[g]/total} for g in groups]})
    allgroups=[g for r in records for g in r['groups']];sizes=[len(g['members']) for g in allgroups]
    metrics={'P0':ev.P,'F0':ev.F,'eta0':ev.E,'communication_edges':len(model.initial_edges),
        'average_degree':2*len(model.initial_edges)/60,'hyperedges_unique':len(ev.hyperedge_counts),
        'task_labeled_groups':len(allgroups),'mean_team_size':float(np.mean(sizes)),
        'mean_groups_per_task':len(allgroups)/50,'zero_group_fraction':float(np.mean([r['effective_groups']==0 for r in records])),
        'feasible_task_fraction':float(ev.feasible_tasks.mean())}
    validation={}
    if pilot:
        reference=model.evaluate(random_inputs=search_inputs(model,index,True,K=1000))
        nodes,edges=model.scfc(ev);ref_nodes,ref_edges=model.scfc(reference)
        tau_node=float(stats.kendalltau(nodes,ref_nodes,variant='b').statistic)
        links=model.initial_edges
        tau_edge=float(stats.kendalltau([edges[e] for e in links],[ref_edges[e] for e in links],variant='b').statistic)
        validation={'K300_K1000_node_tau':tau_node,'K300_K1000_edge_tau':tau_edge,
            'pass_P':ev.P>=.7,'pass_size':5<=metrics['mean_team_size']<=6,
            'pass_node_tau':tau_node>=.9,'pass_edge_tau':tau_edge>=.9}
    return {'n':60,'instance':index,'pilot':pilot,'scene_seed':seeds(60,index,pilot)[0],
        'search_seed':seeds(60,index,pilot)[1],'config':model.config.to_dict(),'positions':model.positions,
        'roles':model.roles,'edges':model.initial_edges,'generation':generation,'metrics':metrics,'tasks':records,'validation':validation}


def compute(count,pilot,workers):
    data=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,index,pilot) for index in range(count)]
        for j,f in enumerate(as_completed(futures),1):
            data.append(f.result())
            if j%10==0 or j==count:print(f'E1 {"validation" if pilot else "formal"}: {j}/{count}',flush=True)
    return sorted(data,key=lambda d:d['instance'])


def aggregate(data):
    taskrows=[];hist=[];type_scene=[]
    for d in data:
        index=d['instance']
        for r in d['tasks']:
            keys=['task','type','effective_groups','successful_searches','P','F','J','top1_share','max_member_share',
                'bottleneck_member_count','has_bottleneck','has_unavoidable_member',
                'bottleneck_S_count','bottleneck_D_count','bottleneck_I_count']
            taskrows.append({'instance':index,**{k:r[k] for k in keys}})
        for typ in range(5):
            rs=[r for r in d['tasks'] if r['type']==typ];sizes=[v for r in rs for v in r['sizes']]
            gen=[g for g in d['generation'] if g['type']==typ]
            rejected=sum(g['rejected'] for g in gen);attempts=sum(g['attempted'] for g in gen)
            type_scene.append({'instance':index,'type':typ,'position_rejection':rejected/attempts,
                'attempted_positions':attempts,'rejected_positions':rejected,
                'P':float(np.mean([r['P'] for r in rs])),
                'mean_team_size':float(np.mean(sizes)) if sizes else None,
                'mean_groups':float(np.mean([r['effective_groups'] for r in rs])),
                'bottleneck_task_fraction':float(np.mean([r['has_bottleneck'] for r in rs])),
                'unavoidable_task_fraction':float(np.mean([r['has_unavoidable_member'] for r in rs]))})
            counts=[sum(v==b if b<9 else v>=9 for v in sizes) for b in range(3,10)]
            overlaps=np.sum([r['overlap_hist'] for r in rs],axis=0)
            for metric,nums in [('size',counts),('overlap',overlaps)]:
                denominator=sum(nums)
                for j,num in enumerate(nums):
                    hist.append({'instance':index,'type':typ,'metric':metric,'bin':j+3 if metric=='size' else j,
                        'count':int(num),'denominator':int(denominator),'proportion':num/denominator if denominator else None})
    bytype=[]
    for typ in range(5):
        chosen=[r for r in type_scene if r['type']==typ];tasks=[r for d in data for r in d['tasks'] if r['type']==typ]
        row={'type':typ,'task_name':TASKS[typ],'K':300,'task_count':len(tasks)}
        for metric in ['position_rejection','P','mean_team_size','mean_groups','bottleneck_task_fraction','unavoidable_task_fraction']:
            mu,ci,n=mean_ci([r[metric] for r in chosen]);row.update({metric:mu,metric+'_ci95':ci,metric+'_n':n})
        row.update(median_groups=med([r['effective_groups'] for r in tasks]),median_J=med([r['J'] for r in tasks]),
            median_top1_share=med([r['top1_share'] for r in tasks]),J_defined_tasks=sum(r['J'] is not None for r in tasks),
            zero_group_tasks=sum(r['effective_groups']==0 for r in tasks))
        attempts=sum(r['attempted_positions'] for r in chosen);rejected=sum(r['rejected_positions'] for r in chosen)
        row.update(attempted_positions=attempts,rejected_positions=rejected,pooled_position_rejection=rejected/attempts)
        total=sum(r['bottleneck_member_count'] for r in tasks)
        for role in 'SDI':
            count=sum(r[f'bottleneck_{role}_count'] for r in tasks)
            row.update({f'bottleneck_{role}_count':count,f'bottleneck_{role}_fraction':count/total if total else None})
        bytype.append(row)
    table={'scenario':'S2','N':60,'n_s':24,'n_d':12,'n_i':24,'M':50,'L':100,'T':120,
        'rS':30,'rc':25,'rI':20,'K':300,'lambda':10,'max_hops':12,'instances':len(data)}
    for metric in data[0]['metrics']:
        mu,ci,n=mean_ci([d['metrics'][metric] for d in data]);table.update({metric:mu,metric+'_ci95':ci,metric+'_n':n})
    write_csv('table3_scenarios.csv',[table]);write_csv('table3_task_types.csv',bytype)
    write_csv('tasks.csv',taskrows);write_csv('histogram_source.csv',hist)
    return table,bytype,hist


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    for suffix in ['', ' Bold', ' Italic', ' Bold Italic']:
        font_manager.fontManager.addfont(str(Path('/System/Library/Fonts/Supplemental')/f'Times New Roman{suffix}.ttf'))
    matplotlib.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
        'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
    import matplotlib.pyplot as plt
    plt.rcParams['mathtext.fontset'] = 'stix'
    return plt


def export(fig,name,axes=None,labels=None):
    from UAV.E2_识别准确性.run import math_notation
    math_notation(fig)
    sys.path.insert(0,'/Users/jackpan/.codex/skills/nature-figure/scripts')
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa=Path('/private/tmp/uav-v11-qa');qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=labels,
        row_groups=[{'id':'comparable-panels','panels':labels}],
        json_out=str(qa/(name+'.alignment.json')),strict=True)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.04);plotting().close(fig)


def select_task(d):
    if 'selected_task' in d:
        return d['tasks'][d['selected_task']],d['tasks'][d['second_task']],d['zoom_bounds']
    pos=np.array(d['positions']);candidates=[]
    for t in d['tasks']:
        if not all(30<=z<=70 for z in t['position']) or len(t['groups'])<8:continue
        members=sorted({i for g in t['groups'][:3] for i in g['members']})
        points=np.vstack([pos[members],t['position']]);span=np.ptp(points,axis=0)
        if max(span)>44:continue
        union=len({i for g in t['groups'][:8] for i in g['members']})
        candidates.append((union,float(np.linalg.norm(np.array(t['position'])-50)),t['task']))
    if not candidates:raise RuntimeError('固定实例0没有合适的中心任务，不能虚构团队')
    task=d['tasks'][min(candidates)[2]]
    other=min([t for t in d['tasks'] if t['task']!=task['task'] and all(30<=z<=70 for z in t['position'])],
        key=lambda t:(-np.linalg.norm(np.array(t['position'])-task['position']),t['task']))
    members=sorted({i for g in task['groups'][:3] for i in g['members']});pts=np.vstack([pos[members],task['position']])
    center=(pts.min(0)+pts.max(0))/2
    center=np.clip(center,25,75);bounds=(center[0]-25,center[0]+25,center[1]-25,center[1]+25)
    return task,other,bounds




def figure4_input(index=0,focus=28):
    model,_=screened_model(60,index);walks=[]
    def on_success(k,r,path):
        if k==focus:walks.append(list(path))
    model.evaluate(random_inputs=search_inputs(model,index),on_success=on_success)
    return {'nodes':{i:(str(model.roles[i]),float(x),float(y)) for i,(x,y) in enumerate(model.positions)},
        'edges':model.initial_edges,'tasks':{k:tuple(t.position) for k,t in enumerate(model.tasks)},
        'focus':focus,'searches':walks,'r_s':30,'r_i':20,'r_c':25,'area':100,'K':300,'task_label':f'T{focus}'}


def rounded_team_field(positions,members,bounds):
    """节点圆盘与绕开非成员的连通管道形成圆角集合轮廓，不表示物理链路。"""
    from heapq import heappush,heappop
    from scipy.spatial.distance import cdist
    from scipy.sparse.csgraph import minimum_spanning_tree
    x0,x1,y0,y1=bounds;step=.30
    x=np.arange(x0,x1+step/2,step);y=np.arange(y0,y1+step/2,step);xx,yy=np.meshgrid(x,y)
    outside=[i for i in range(len(positions)) if i not in members and x0-2<=positions[i,0]<=x1+2 and y0-2<=positions[i,1]<=y1+2]
    distance=np.full(xx.shape,np.inf)
    for i in outside:distance=np.minimum(distance,np.hypot(xx-positions[i,0],yy-positions[i,1]))
    blocked=distance<1.55;mask=np.zeros_like(blocked)
    def cell(p):return tuple(np.clip(np.round([(p[1]-y0)/step,(p[0]-x0)/step]).astype(int),[0,0],np.array(mask.shape)-1))
    cells=[cell(positions[i]) for i in members]
    for r,c in cells:mask[r,c]=True;blocked[r,c]=False
    points=positions[members];tree=minimum_spanning_tree(cdist(points,points)).tocoo()
    for a,b in zip(tree.row,tree.col):
        start,goal=cells[a],cells[b];queue=[(0.,0.,start)];cost={start:0.};parent={};found=False
        while queue:
            _,g,current=heappop(queue)
            if g>cost[current]+1e-9:continue
            if current==goal:found=True;break
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr,nc=current[0]+dr,current[1]+dc
                if not(0<=nr<mask.shape[0] and 0<=nc<mask.shape[1]) or blocked[nr,nc]:continue
                nxt=(nr,nc);new=g+math.hypot(dr,dc)*(1+.6/max(distance[nr,nc],.3))
                if new<cost.get(nxt,np.inf):
                    cost[nxt]=new;parent[nxt]=current
                    heappush(queue,(new+math.hypot(nr-goal[0],nc-goal[1]),new,nxt))
        if not found:raise RuntimeError('成员轮廓避让路径不可达')
        cursor=goal
        while cursor!=start:mask[cursor]=True;cursor=parent[cursor]
        mask[start]=True
    radius=1.8;offset=int(np.ceil(radius/step));dy,dx=np.mgrid[-offset:offset+1,-offset:offset+1]
    field=ndimage.gaussian_filter(ndimage.binary_dilation(mask,structure=(dx*dx+dy*dy<=(radius/step)**2)).astype(float),1.)
    # 非成员周围保留空洞；接近成员时自适应缩小洞，避免吞掉真实成员。
    for i in outside:
        radius=min(1.55,float(np.min(np.linalg.norm(points-positions[i],axis=1)))*.4)
        field[np.hypot(xx-positions[i,0],yy-positions[i,1])<=radius]=0
    for i in members:
        radius=min(.8,float(np.min(np.linalg.norm(positions[outside]-positions[i],axis=1)))*.3) if outside else .8
        field[np.hypot(xx-positions[i,0],yy-positions[i,1])<=radius]=1
    from scipy.interpolate import RegularGridInterpolator
    sample=RegularGridInterpolator((y,x),field,bounds_error=False,fill_value=0.)
    assert np.all(sample(points[:,::-1])>.5)
    assert not outside or np.all(sample(positions[outside,::-1])<.5)
    return x,y,field


def place_label(ax,text,anchor,color):
    fig=ax.figure;fig.canvas.draw();renderer=fig.canvas.get_renderer()
    paths=[artist.get_path().transformed(artist.get_transform()) for artist in [*ax.lines,*ax.patches] if not artist.get_rasterized()]
    nodes=[]
    for collection in ax.collections:
        offsets=np.asarray(collection.get_offsets())
        if not len(offsets):continue
        points=collection.get_offset_transform().transform(offsets);sizes=collection.get_sizes()
        radius=(np.sqrt(np.max(sizes))/2 if len(sizes) else 2)+3
        nodes.append((points,radius*fig.dpi/72))
    previous=[t.get_window_extent(renderer) for t in ax.texts]
    label=ax.text(*anchor,text,ha='center',va='center',color=color,fontsize=8,weight='bold' if text.startswith('T') else 'normal',zorder=6)
    center=ax.transData.transform(anchor);frame=ax.get_window_extent(renderer).padded(-2);options=[]
    for radius in [8,11,14,17,20,24,28,32,36]:
        for angle in range(0,360,20):
            pixel=center+radius*fig.dpi/72*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            xy=ax.transData.inverted().transform(pixel);label.set_position(xy)
            box=label.get_window_extent(renderer).padded(3*fig.dpi/72)
            if not frame.contains(box.x0,box.y0) or not frame.contains(box.x1,box.y1):continue
            strokes=sum(p.intersects_bbox(box,filled=False) for p in paths)
            hits=sum(box.overlaps(p) for p in previous)
            markers=sum(np.sum((p[:,0]>=box.x0-r)&(p[:,0]<=box.x1+r)&(p[:,1]>=box.y0-r)&(p[:,1]<=box.y1+r)) for p,r in nodes)
            options.append(((strokes*100+hits*200+markers*20,radius,angle),xy))
    if not options:raise RuntimeError('无法放置标签')
    score,xy=min(options,key=lambda item:item[0]);label.set_position(xy)
    return {'text':text,'position':xy.tolist(),'clearance_score':score[0]}


def figure4_legacy_scene(scene):
    if 'positions' in scene:return scene
    nodes={int(i):v for i,v in scene['nodes'].items()}
    counter=Counter(frozenset(path) for path in scene['searches'])
    n=sum(counter.values());focus=int(scene['focus'])
    groups=[{'members':sorted(g),'count':int(c),'share':c/n}
        for g,c in sorted(counter.items(),key=lambda item:(-item[1],tuple(sorted(item[0]))))[:8]]
    tasks=[{'task':int(k),'position':v,'groups':groups if int(k)==focus else []}
        for k,v in sorted(scene['tasks'].items(),key=lambda item:int(item[0]))]
    d={'instance':0,'positions':[nodes[i][1:] for i in sorted(nodes)],
        'roles':[nodes[i][0] for i in sorted(nodes)],'edges':scene['edges'],'tasks':tasks}
    task,other,bounds=select_task(d)
    d.update(selected_task=task['task'],second_task=other['task'],zoom_bounds=bounds)
    return d


def render_scenario(data):
    plt=plotting()
    plt.rcParams['mathtext.fontset'] = 'stix'
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle,Arc
    from matplotlib.colors import to_rgba
    d=figure4_legacy_scene(data[0]);pos=np.array(d['positions']);roles=np.array(d['roles']);task,other,bounds=select_task(d)
    fig=plt.figure(figsize=(190/25.4,95/25.4))
    a=fig.add_axes([.055,.22,.265,.56]);b=fig.add_axes([.375,.22,.265,.56])
    matrix=fig.add_axes([.735,.22,.205,.41]);top=fig.add_axes([.735,.65,.205,.10]);side=fig.add_axes([.95,.22,.035,.41])
    for ax in [a,b]:
        for u,v in d['edges']:ax.plot(pos[[u,v],0],pos[[u,v],1],color='#777777',alpha=.3 if ax is a else .15,lw=.45,zorder=1,rasterized=True)
        ax.set_aspect('equal',adjustable='box');ax.tick_params(length=2,pad=2)
        for spine in ax.spines.values():spine.set_visible(True)
    for role in 'SDI':
        p=pos[roles==role];a.scatter(p[:,0],p[:,1],marker=MARKERS[role],s=13,color=ROLE_COLORS[role],edgecolors='white',lw=.2,zorder=4)
    a.scatter([t['position'][0] for t in d['tasks']],[t['position'][1] for t in d['tasks']],marker='x',s=9,color='#CF7375',alpha=.5,lw=.6,zorder=3,rasterized=True)
    for t in [task,other]:
        p=t['position'];a.scatter(*p,marker='x',s=25,color='#A92F39',lw=1,zorder=5)
        for radius,color,style in [(30,ROLE_COLORS['S'],'--'),(20,ROLE_COLORS['I'],':')]:
            assert all(radius<=z<=100-radius for z in p)
            a.add_patch(plt.Circle(p,radius,fill=False,color=color,ls=style,lw=.65,zorder=2))
    x0,x1,y0,y1=bounds;a.add_patch(Rectangle((x0,y0),50,50,fill=False,edgecolor='#555555',ls='--',lw=.7,zorder=2))
    a.plot([73,98],[2,2],color='#222222',lw=1.2,zorder=5)
    a.set(xlim=(0,100),ylim=(0,100),xlabel='X',ylabel='Y',xticks=[0,50,100],yticks=[0,50,100])
    spatial_labels=[place_label(a,f'T{t["task"]}',t['position'],'#A92F39') for t in [task,other]]
    for label in a.texts:
        if label.get_text()=='T38':label.set_position((39,45.5))
    for label in spatial_labels:
        if label['text']=='T38':label.update(position=[39,45.5],clearance_score='final PDF checked')
    a.text(85.5,6.5,'rc = 25',ha='center',va='center',fontsize=8)
    a.text(x0+1,y1-1,'(b)',ha='left',va='top',fontsize=8,color='#555555')
    selected=[]
    for rank,g in enumerate(task['groups'],1):
        members_set=set(g['members'])
        if all(not(members_set<=set(other['members']) or set(other['members'])<=members_set) for _,other in selected):
            selected.append((rank,g))
        if len(selected)==2:break
    assert len(selected)==2
    team_colors={rank:color for (rank,_),color in zip(selected,TEAM_COLORS)}
    union=sorted({i for _,g in selected for i in g['members']})
    count=Counter(i for _,g in selected for i in g['members']);shared=[i for i,c in count.items() if c>1]
    paths=[]
    for j,(rank,g) in enumerate(selected):
        x,y,field=rounded_team_field(pos,g['members'],bounds)
        rgba=np.zeros((*field.shape,4));rgba[:,:,:3]=to_rgba(TEAM_COLORS[j])[:3];rgba[:,:,3]=.15*(field>.5)
        b.imshow(rgba,origin='lower',extent=(x[0],x[-1],y[0],y[-1]),interpolation='bilinear',zorder=2)
        b.contour(x,y,field,levels=[.5],colors=[TEAM_COLORS[j]],linewidths=[.9],zorder=2.2)
        paths.append({'rank':rank,'members':g['members'],'share':g['share']})
    for radius,color,style in [(30,ROLE_COLORS['S'],'--'),(20,ROLE_COLORS['I'],':')]:
        # Export only visible arcs so the vector geometry stays inside the zoom frame.
        cx,cy=task['position'];angles={0.,2*np.pi}
        for boundary in [x0,x1]:
            value=(boundary-cx)/radius
            if abs(value)<=1:
                angle=float(np.arccos(value));angles.update([angle,2*np.pi-angle])
        for boundary in [y0,y1]:
            value=(boundary-cy)/radius
            if abs(value)<=1:
                angle=float(np.arcsin(value));angles.update([angle%(2*np.pi),(np.pi-angle)%(2*np.pi)])
        angles=sorted(angles)
        for start,end in zip(angles,angles[1:]):
            mid=(start+end)/2;px=cx+radius*np.cos(mid);py=cy+radius*np.sin(mid)
            if end-start>1e-10 and x0<=px<=x1 and y0<=py<=y1:
                b.add_patch(Arc((cx,cy),2*radius,2*radius,theta1=np.degrees(start),theta2=np.degrees(end),
                    color=color,ls=style,lw=.65,zorder=1.8))
    visible=(pos[:,0]>=x0)&(pos[:,0]<=x1)&(pos[:,1]>=y0)&(pos[:,1]<=y1)
    for role in 'SDI':
        p=pos[(roles==role)&visible];b.scatter(p[:,0],p[:,1],marker=MARKERS[role],s=15,color='#D1D4D6',edgecolors='white',lw=.2,zorder=3)
        ids=[i for i in union if roles[i]==role];b.scatter(pos[ids,0],pos[ids,1],marker=MARKERS[role],s=19,color=ROLE_COLORS[role],edgecolors='white',lw=.3,zorder=4)
    b.scatter(pos[shared,0],pos[shared,1],s=35,facecolors='none',edgecolors='#222222',lw=.7,zorder=4.5)
    b.scatter(*task['position'],marker='x',s=25,color='#A92F39',lw=1,zorder=5)
    b.text(task['position'][0]+1,task['position'][1]+2,f'T{task["task"]}',fontsize=8,color='#A92F39',weight='bold',zorder=6)
    # 直接标注所显示的两条轮廓；具体坐标随数据检查记录保存。
    labels=[]
    for j,(rank,g) in enumerate(selected):
        xy=[(x1-13,y0+20),(x1-13,y1-4)][j]
        text=rf'$e_{{{rank}}}$: {g["share"]:.1%}'
        b.text(*xy,text,ha='center',va='center',color=TEAM_COLORS[j],fontsize=8,zorder=6)
        labels.append({'text':text,'position':xy})
    b.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='X',ylabel='Y')
    from matplotlib.ticker import MaxNLocator
    b.xaxis.set_major_locator(MaxNLocator(3));b.yaxis.set_major_locator(MaxNLocator(3))
    eight=task['groups'][:8];members=sorted({i for g in eight for i in g['members']},key=lambda i:('SDI'.index(roles[i]),i))
    yy=[];last=None;cursor=0
    for i in members:
        if last is not None and last!=roles[i]:cursor+=.8
        yy.append(cursor);cursor+=1;last=roles[i]
    links=[]
    for j,g in enumerate(eight):
        ys=[yy[i] for i,node in enumerate(members) if node in g['members']];color=team_colors.get(j+1,'#555555')
        matrix.plot([j,j],[min(ys),max(ys)],color=color,lw=.65,zorder=1)
        matrix.scatter([j]*len(ys),ys,color=color,s=14,zorder=2)
        links.extend({'rank':j+1,'node':node,'role':roles[node],'present':node in g['members']} for node in members)
    matrix.set(xlim=(-.65,7.65),ylim=(max(yy)+.7,-.7),xticks=range(8),xticklabels=[rf'$e_{{{i}}}$' for i in range(1,9)],yticks=yy,
        yticklabels=[f'{roles[i]}{i}' for i in members],xlabel='Team')
    matrix.tick_params(length=0,pad=2)
    for spine in matrix.spines.values():spine.set_visible(False)
    top.bar(range(8),[g['share'] for g in eight],color=[team_colors.get(i+1,'#555555') for i in range(8)],width=.65)
    top.set(xlim=matrix.get_xlim(),xticks=[],ylabel='Share')
    limit=math.ceil(max(g['share'] for g in eight)*1.15/.01)*.01
    top.set(ylim=(0,limit),yticks=[0,limit],yticklabels=['0%',f'{100*limit:g}%'])
    side.barh(yy,[sum(i in g['members'] for g in eight) for i in members],height=.6,color='#B1B6BA')
    side.set(ylim=matrix.get_ylim(),xlim=(0,11),xticks=[],yticks=[])
    side.patch.set_visible(False)
    for y,i in zip(yy,members):side.text(sum(i in g['members'] for g in eight)+1.4,y,str(sum(i in g['members'] for g in eight)),va='center',fontsize=8)
    for spine in side.spines.values():spine.set_visible(False)
    # Keep all three titles on one baseline, close to the upper content boundary.
    title_y=a.get_position().y1+6/(fig.get_figheight()*72)
    for ax,title in [(a,'(a) Communication network'),(b,'(b) Alternative teams'),(top,'(c) Team membership')]:
        fig.text(ax.get_position().x0,title_y,title,ha='left',va='bottom',fontsize=8.5,weight='bold')
    handles=[Line2D([],[],marker=MARKERS[r],color='none',markerfacecolor=ROLE_COLORS[r],label=r,markersize=4) for r in 'SDI']
    handles += [Line2D([],[],marker='x',color='#CF7375',ls='',label='Task',markersize=4),Line2D([],[],color='#D6D6D6',lw=.6,label='Link'),
        Line2D([],[],color=ROLE_COLORS['S'],ls='--',label='Sensing radius 30'),Line2D([],[],color=ROLE_COLORS['I'],ls=':',label='Execution radius 20'),
        Line2D([],[],marker='o',markerfacecolor='none',color='#222222',ls='',label='Shared member',markersize=4)]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.51,.005),frameon=False,columnspacing=1.1,handlelength=1.4)
    export(fig,'fig04_scenario',[a,b],['a','b'])


def render_structure(data,hist):
    plt=plotting();fig=plt.figure(figsize=(190/25.4,78/25.4))
    plt.rcParams['mathtext.fontset'] = 'stix'
    # Equal-width panels keep the three structural summaries directly readable.
    axes=[fig.add_axes([left,.28,.25,.55]) for left in [.075,.405,.735]]
    tasks=[r for d in data for r in d['tasks']]
    for col,ax in enumerate(axes):
        ax.set_title(['(a) Team size','(b) Teams and bottlenecks','(c) Team overlap'][col],
            loc='left',fontsize=8.5,weight='bold',pad=10)
        ax.tick_params(length=2,pad=2)
        if col in [0,2]:
            metric='size' if col==0 else 'overlap';bins=list(range(3,10)) if col==0 else list(range(4));lower=0.;upper=0.
            for typ,color in enumerate(COLORS):
                values=[mean_ci([r['proportion'] for r in hist if r['type']==typ and r['metric']==metric and r['bin']==b]) for b in bins]
                means=np.array([v[0] for v in values]);ci=np.array([v[1] for v in values])
                lower=min(lower,float(np.min(means-ci)));upper=max(upper,float(np.max(means+ci)))
                ax.bar(np.arange(len(bins))+(typ-2)*.15,means,width=.14,color=color,yerr=ci,
                    error_kw={'lw':.5,'capsize':1,'capthick':.5})
            top=math.ceil((upper+.025)/.05)*.05
            ax.set(ylim=(min(0,lower-.005) if lower<0 else 0,top),ylabel='Proportion',
                xlabel='Team size |e|' if col==0 else 'Shared S/D members',xticks=range(len(bins)),
                xticklabels=['3','4','5','6','7','8','≥9'] if col==0 else ['0','1','2','≥3'])
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(4))
        else:
            order=np.random.default_rng(20261026).permutation(len(tasks))
            points=[tasks[int(i)] for i in order]
            ax.scatter([r['effective_groups'] for r in points],[r['max_member_share'] for r in points],
                c=[COLORS[r['type']] for r in points],s=7,alpha=.38,edgecolors='none',rasterized=True)
            ax.axhline(.9,color='#555555',ls='--',lw=.7)
            ax.annotate('90%',xy=(1,.9),xycoords=('axes fraction','data'),xytext=(3,0),textcoords='offset points',
                ha='left',va='center',fontsize=8,color='#555555',clip_on=False)
            ax.set(xlim=(0,300),ylim=(0,1.03),xticks=[0,100,200,300],yticks=[0,.5,1],
                xlabel='Distinct teams per task',ylabel='Max. member share')
            ax.tick_params(axis='x',length=2,pad=2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(COLORS,TASKS)],loc='lower center',
        bbox_to_anchor=(.53,.085),ncol=5,frameon=False,columnspacing=1.2,handlelength=1.2)
    export(fig,'fig05_structure',list(axes),list('abc'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--instances',type=int,default=50);p.add_argument('--pilot-instances',type=int,default=10)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--K',type=int,default=300)
    p.add_argument('--plot-only',action='store_true');p.add_argument('--compute-only',action='store_true');p.add_argument('--force',action='store_true')
    args=p.parse_args()
    if args.force and not args.plot_only and OUT.exists():shutil.rmtree(OUT)
    OUT.mkdir(exist_ok=True)
    if args.K!=300:raise ValueError('E1固定K=300')
    if args.plot_only:plot();return
    data=compute(args.instances,False,args.workers);pilot=compute(args.pilot_instances,True,args.workers)
    write_json(OUT/'run_config.json',{'design':'v11','SEED':SEED,'baseline':config().to_dict(),
        'instances':args.instances,'independent_validation_instances':args.pilot_instances,
        'feasible_task_screening':True,'bottleneck_denominator':'successful searches per task',
        'bottleneck_role_denominator':'all bottleneck task-member incidences within each task type',
        'source_model_sha256':hashlib.sha256((UAV/'model.py').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((UAV/'experiment_config.py').read_bytes()).hexdigest()})
    table,bytype,hist=aggregate(data)
    write_csv('independent_validation.csv',[{'instance':d['instance'],**d['metrics'],**d['validation']} for d in pilot])
    validation_summary=[]
    for key,target in [('P0',.7),('mean_team_size',None),('K300_K1000_node_tau',.9),('K300_K1000_edge_tau',.9)]:
        vals=[d['metrics'].get(key,d['validation'].get(key)) for d in pilot];mu,ci,n=mean_ci(vals)
        validation_summary.append({'metric':key,'mean':mu,'ci95':ci,'instances':n,
            'target':'5–6' if target is None else f'>={target}',
            'pass_mean':5<=mu<=6 if target is None else mu>=target})
    write_csv('independent_validation_summary.csv',validation_summary)
    print('FORMAL',json.dumps(table,ensure_ascii=False),flush=True)
    print('BY_TYPE',json.dumps(bytype,ensure_ascii=False),flush=True)
    print('VALIDATION',json.dumps(validation_summary,ensure_ascii=False),flush=True)
    d=data[0];task,_,_=select_task(d)
    scene=figure4_input(d['instance'],task['task'])
    write_json(OUT/'scenario_plot.json',scene)
    if not args.compute_only:plot()

def plot():
    scene=json.loads((OUT/'scenario_plot.json').read_text())
    tasks=read_csv(OUT/'tasks.csv')
    render_scenario([scene]);render_structure([{'tasks':tasks}],read_csv(OUT/'histogram_source.csv'))
    write_captions(tasks,scene)


def write_captions(tasks,scene):
    high=[t for t in tasks if t['effective_groups']>100];bottleneck=[t for t in high if t['max_member_share']>=.9]
    rows=[{'图表':'图4','图注或表注':'基准场景0及任务T28的备选团队。'
        '(a)通信网络按S/D/I角色编码，显示T28、T38的侦察半径30与执行半径20，虚线框表示(b)的50×50局部视窗。'
        '(b)同时绘制T28的侦察半径30与执行半径20，颜色、线型与(a)一致，视窗外的圆弧按局部边界裁切。按成功搜索份额降序扫描团队，只保留与已选团队两两互不包含的前两条超边；有包含关系则顺延。本实例选出e1、e2，分别用紫、青色轮廓表示；旁注原始排名及份额，黑圈表示选定团队中的共享成员。'
        '(c)列为份额最高的前8个团队，上方柱状图为各团队占成功搜索的份额；矩阵显示团队成员，仅e1、e2两列及列顶份额柱使用与(b)相同的紫、青色，其余列为深灰。右侧数字及条形表示该成员在前8个团队中出现的次数，范围0–8，并非全部成功搜索中的覆盖率。'
        '团队按份额降序排列，同频时按成员编号字典序排列；通信比例尺长25，标注位于图框内。',
        '建议正文句':''},
        {'图表':'图5','图注或表注':f'50个独立S2场景、2500个任务的协同超图统计，K=300；所有任务均至少找到一个团队。'
        '(a)同一场景内、同一任务类型下按不同团队计数的团队规模比例，柱为50个场景的均值，误差棒为场景级95% Student-t置信区间。'
        '(b)每点为一个任务，横轴为K=300次搜索发现的不同成员集合数，纵轴为某成员在该任务全部成功搜索中的最高出现比例，颜色区分任务类型；虚线为90%瓶颈阈值。'
        f'团队数>100的{len(high)}个任务中，{len(bottleneck)}个（{len(bottleneck)/len(high):.1%}）仍存在瓶颈成员。'
        '(c)同任务不同团队对共享S/D成员数的场景内比例；柱为50个场景均值，误差棒为场景级95% Student-t置信区间。重叠数A未经团队规模归一化。',
        '建议正文句':'较大的团队具有更多可共享成员，因此Composite的原始重叠人数偏高可能部分反映团队规模差异，不能全部归因于额外的协同耦合；此处保留SCFC直接使用的原始重叠数A。'},
        {'图表':'表3','图注或表注':'“存在瓶颈成员的任务比例（某成员覆盖 ≥90% 成功搜索）”指：对每个任务，将包含成员i的全部团队出现次数相加，再除以该任务成功搜索总次数，得到p_ki；若max_i p_ki≥0.9，则该任务计为有瓶颈成员。'
        '统计按每场景每类型10个任务先求比例，再对50个独立场景求均值与95% Student-t置信区间。'
        '模型中的节点任务不可替代度c_i=(1/M)Σ_k 1[i属于该任务全部已发现有效团队的交集]，对应p_ki=1的严格情形；表中≥90%是更宽松的任务级近似瓶颈阈值，不等同于c_i，也不保证不存在稀有替代团队。所有判断均基于有限K搜索发现的团队。瓶颈成员的角色构成按任务—成员计数汇总，同一节点在不同任务中分别计数；S/D/I比例之和为1。',
        '建议正文句':''}]
    write_csv('captions.csv',rows)


def read_csv(path):
    def value(v):
        if v=='':return None
        if v in ['True','False']:return v=='True'
        try:return int(v)
        except ValueError:
            try:return float(v)
            except ValueError:return v
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return [{k:value(v) for k,v in row.items()} for row in csv.DictReader(f)]

if __name__=='__main__':main()

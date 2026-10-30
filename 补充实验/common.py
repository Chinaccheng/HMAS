"""Read-only supplemental experiment helpers. All statistics use scene-level samples."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']='/private/tmp/uav-supplement-numba'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from UAV.experiment_common import *
import csv,hashlib,platform,subprocess,zipfile,tempfile
from collections import Counter
HERE=Path(__file__).resolve().parent
OUT=HERE/'result';OUT.mkdir(exist_ok=True)
# Share cached jobs only with subprocesses in this run. A fixed cache can
# silently reuse results produced by a different model, configuration, or run.
_cache_dir = None
if 'UAV_SUPPLEMENT_CACHE_DIR' not in os.environ:
    _cache_dir = tempfile.TemporaryDirectory(prefix='uav-supplement-')
    os.environ['UAV_SUPPLEMENT_CACHE_DIR'] = _cache_dir.name
CACHE=Path(os.environ['UAV_SUPPLEMENT_CACHE_DIR']);CACHE.mkdir(parents=True,exist_ok=True)
SNAPSHOT=CACHE/'original_hashes.json'
DIRS={i:next(ROOT.glob(f'E{i}_*'))/'result' for i in range(1,8)}
def read(e,name):return read_csv(DIRS[e]/name)
def save(name,rows):
    rows=list(rows)
    if not rows:raise ValueError(f'No rows: {name}')
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    print(f'{name}: {len(rows)} rows',flush=True)
    return rows

def stat(values,prefix):
    mu,ci,n=mean_ci(values);return {prefix:mu,prefix+'_ci95':ci,prefix+'_n':n}

def test(a,b,label='difference'):
    ids=sorted(a.keys()&b.keys());v=np.array([a[i]-b[i] for i in ids if a[i] is not None and b[i] is not None],float)
    return {**stat(v,label),'p_two_sided':float(stats.wilcoxon(v,alternative='two-sided').pvalue) if len(v) and np.any(v) else (1. if len(v) else None)}

def compare(rows,metric,baseline,**keys):
    a={r['instance']:r[metric] for r in rows if r['method']=='SCFC'}
    b={r['instance']:r[metric] for r in rows if r['method']==baseline}
    return {**keys,'metric':metric,'comparison':'SCFC - '+baseline,'baseline':baseline,**test(a,b)}

def best_method(rows,metric,exclude=('SCFC','random','greedy'),higher=True):
    methods=sorted({r['method'] for r in rows}-set(exclude))
    means={m:mean_ci([r.get(metric) for r in rows if r['method']==m])[0] for m in methods}
    means={m:v for m,v in means.items() if v is not None}
    return (max if higher else min)(means,key=means.get)

def cache_job(name,func,*args):
    path=CACHE/(name+'.json')
    if path.exists():return read_json(path)
    result=func(*args);write_json(path,result);return result

def cohort_objects():
    return [r for r in read(2,'object_scores_and_truth.csv') if r['n']==60]

def snapshot():
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and HERE not in p.parents and '.git' not in p.parts and p.name!='.DS_Store' and '__pycache__' not in p.parts}
    write_json(SNAPSHOT,hashes)
    return hashes

def verify_snapshot():
    before=read_json(SNAPSHOT);changed=[]
    for rel,h in before.items():
        p=ROOT/rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(rel)
    write_json(OUT/'original_files_verification.json',{'checked_files':len(before),'changed_files':changed,'passed':not changed})
    assert not changed,changed

def environment():
    import scipy,numba,networkx
    r={'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'numba':numba.__version__,'networkx':networkx.__version__,
       'logical_cpu':os.cpu_count(),'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'simulation_workers_per_task':1,'process_pool_used':False,'validation_overlap_during_development':True,'C1_exclusive_single_process':True,'numba_actual_threads':numba.get_num_threads()}
    for key,query in [('cpu_model','machdep.cpu.brand_string'),('physical_cpu','hw.physicalcpu'),('memory_bytes','hw.memsize')]:
        result=subprocess.run(['/usr/sbin/sysctl','-n',query],text=True,capture_output=True)
        if result.returncode==0:r[key]=result.stdout.strip() if key=='cpu_model' else int(result.stdout.strip())
    if 'memory_bytes' in r:r['memory_GiB']=r['memory_bytes']/2**30
    write_json(OUT/'environment.json',r)
    return r

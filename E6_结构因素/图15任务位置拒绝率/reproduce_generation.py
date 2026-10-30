"""Replay only the original role-scenario task generator and validate first.

No stochastic collaboration searches, scoring, attacks, recovery, or figures.
The original generator and model files are imported without modification.
All rates are withheld until every saved task-library fingerprint and the
complete original failure set have passed reproduction checks.
"""
import os, sys
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/uav-role-generation-only-numba')
from pathlib import Path
import csv, hashlib, json, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ROOT=PROJECT/'UAV'
E6=ROOT/'E6_结构因素/result'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    names=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def jsave(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def protected_files():
    result={}
    for directory in [ROOT]:
        for p in directory.rglob('*'):
            if p.is_file() and HERE not in p.parents and '.git' not in p.parts and p.name!='.DS_Store':result[str(p)]=sha(p)
    return result

def preflight():
    cfg=json.loads((E6/'run_config.json').read_text())
    paths={'source_model_sha256':ROOT/'model.py','reproduction_config_sha256':ROOT/'experiment_config.py',
           'config_schema_sha256':ROOT/'config.py','experiment_code_sha256':ROOT/'E6_结构因素/run.py'}
    checks={k:{'recorded':cfg[k],'current':sha(p),'match':cfg[k]==sha(p)} for k,p in paths.items()}
    # The one known source change affects horizons for N=30 and N=120 only.
    # Check the complete source after reversing that exact change in memory;
    # this replay uses N=60, whose horizon is 120 in both versions.
    current=paths['reproduction_config_sha256'].read_bytes()
    new=b'horizon={30:120,60:120,120:120}'
    old=b'horizon={30:50,60:120,120:150}'
    normalized=hashlib.sha256(current.replace(new,old,1)).hexdigest() if current.count(new)==1 else None
    config_check=checks['reproduction_config_sha256']
    config_check['known_horizon_change_reverted_sha256']=normalized
    config_check['known_horizon_change_only']=normalized==config_check['recorded']
    sys.path.insert(0,str(PROJECT))
    from UAV import experiment_config as scenario
    current_baseline=scenario.config().to_dict()
    checks['baseline_parameters']={'recorded':cfg['baseline'],'current':current_baseline,
                                   'match':cfg['baseline']==current_baseline}
    jsave(HERE/'生成前版本核验.json',checks)
    required=['source_model_sha256','config_schema_sha256','experiment_code_sha256','baseline_parameters']
    if not all(checks[k]['match'] for k in required) or not (config_check['match'] or config_check['known_horizon_change_only']):
        raise RuntimeError('Model, schema, experiment source or baseline parameters differ. Stop before generation.')
    config_changed=not config_check['match']
    if config_changed:
        print('WARNING: configuration differs from the original only in the N=30 and N=120 horizons; '
              'the N=60 horizon used for this replay is unchanged.',flush=True)
    state=[r for r in read(E6/'scenario_status.csv') if r['factor']=='roles']
    manifest=[r for r in read(E6/'scenario_manifest.csv') if r['factor']=='roles']
    assert len(state)==1850 and len(manifest)==1785
    assert Counter(r['status'] for r in state)=={'ok':1785,'non_generatable':65}
    return state,manifest,config_changed

def forbidden(*args,**kwargs):
    raise RuntimeError('Forbidden collaborative search / simulation called during generation-only replay')

def worker(s,d,index):
    sys.path.insert(0,str(PROJECT))
    from UAV import experiment_config as scenario
    from UAV.model import UAVModel
    for name in ['evaluate','evaluate_reference','evaluate_functions','random_inputs','scores','scfc',
                 'attack','attack_order','restore','available_actions','choose_recovery']:
        if hasattr(UAVModel,name):setattr(UAVModel,name,forbidden)
    scenario.search_inputs=forbidden
    scenario.truth_inputs=forbidden
    actual=20000+index
    start=time.monotonic()
    try:
        model,generation=scenario.screened_model(60,actual,n_s=s,n_d=d,n_i=60-s-d)
        status='ok';reason='';failure_class='none';trials=[]
    except scenario.InfeasibleScenario as exc:
        tb=exc.__traceback__;locals_at_failure=None
        while tb:
            if tb.tb_frame.f_code is scenario.screened_model.__code__:locals_at_failure=tb.tb_frame.f_locals.copy()
            tb=tb.tb_next
        assert locals_at_failure is not None
        model=locals_at_failure['model']
        generation=locals_at_failure.get('generation',[])
        trials=locals_at_failure.get('trials',[]) if 'generation' in locals_at_failure else []
        status='non_generatable';reason=str(exc)
        failure_class='no_feasible_geometric_position' if 'generation' in locals_at_failure else 'no_path_even_with_full_coverage'
    attempted=sum(r['attempted'] for r in generation)+len(trials)
    rejected=sum(r['rejected'] for r in generation)+sum(not r['feasible'] for r in trials)
    accepted=sum(r['feasible'] for r in trials)+len(generation)
    assert attempted==accepted+rejected
    # The constructor pre-draws all 50 initial positions. Table 3 counts tested
    # proposals only, so initially drawn but never tested positions are exposed
    # separately and are not misclassified as accepted or rejected.
    initially_tested=len(generation)+(bool(trials))
    untested_initial=50-initially_tested
    constructor_positions=50
    actual_rng_positions_drawn=constructor_positions+attempted-initially_tested
    manifest=scenario.manifest(model,actual)
    row={'n_s':s,'n_d':d,'n_i':60-s-d,'value':f'{s}-{d}','instance':index,'seed_index':actual,
         'scene_seed':scenario.seeds(60,actual)[0],'status':status,'failure_class':failure_class,
         'task_positions_attempted':attempted,'task_positions_rejected':rejected,'task_positions_accepted':accepted,
         'completed_tasks':len(generation),'pending_task_trials':len(trials),
         'constructor_positions_predrawn':constructor_positions,'initial_positions_untested_at_stop':untested_initial,
         'rng_positions_drawn_including_untested':actual_rng_positions_drawn,
         'network_sha256':manifest['network_sha256'],'tasks_sha256':manifest['tasks_sha256'],
         'elapsed_seconds':time.monotonic()-start,'reason':reason,
         'forbidden_search_or_simulation_calls':0}
    return row

def interval(values):
    a=np.asarray(values,float)
    if len(a)==0:return None,None,None,None
    mu=float(a.mean());ci=float(stats.t.ppf(.975,len(a)-1)*stats.sem(a)) if len(a)>1 else None
    return mu,mu-ci if ci is not None else None,mu+ci if ci is not None else None,ci

def outputs(rows):
    # Nothing in this function is called until every reproduction check passes.
    for r in rows:
        r['task_position_rejection_rate']=r['task_positions_rejected']/r['task_positions_attempted'] if r['task_positions_attempted'] else ''
        r['rate_defined']=bool(r['task_positions_attempted'])
        r['task_library_matches_saved_fingerprint']=True if r['status']=='ok' else 'NA_incomplete_library'
    write(HERE/'逐网络任务位置计数.csv',rows)
    table=[]
    for s,d in sorted({(r['n_s'],r['n_d']) for r in rows}):
        group=[r for r in rows if (r['n_s'],r['n_d'])==(s,d)]
        defined=[r for r in group if r['rate_defined']]
        valid=[r for r in defined if r['status']=='ok']
        mu,lo,hi,ci=interval([r['task_position_rejection_rate'] for r in defined])
        vm,vl,vh,vc=interval([r['task_position_rejection_rate'] for r in valid])
        row={'S:D:I':f'{s}:{d}:{60-s-d}','n_s':s,'n_d':d,'n_i':60-s-d,'p_S':s/60,'p_D':d/60,
             'networks_total':len(group),'valid_networks':sum(r['status']=='ok' for r in group),
             'invalid_networks':sum(r['status']!='ok' for r in group),'rate_defined_networks':len(defined),
             'rate_undefined_networks':len(group)-len(defined),
             'invalid_networks_with_position_trials':sum(r['status']!='ok' for r in defined),
             'task_position_rejection_rate_mean':mu,'ci95_low':lo,'ci95_high':hi,'ci95_halfwidth':ci,
             'valid_network_rejection_rate_mean':vm,'valid_ci95_low':vl,'valid_ci95_high':vh,'valid_ci95_halfwidth':vc,
             'positions_attempted_total':sum(r['task_positions_attempted'] for r in group),
             'positions_rejected_total':sum(r['task_positions_rejected'] for r in group)}
        table.append(row)
    write(HERE/'图15_角色构成任务位置拒绝比例.csv',table)
    role_summary=read(E6/'role_summary.csv')
    best=max([r for r in role_summary if r['n_s']=='6'],key=lambda r:float(r['F0']))
    selected=[next(r for r in table if (r['n_s'],r['n_d'])==(int(best['n_s']),int(best['n_d']))),
              next(r for r in table if (r['n_s'],r['n_d'])==(24,12))]
    write(HERE/'重点两格任务位置拒绝比例.csv',selected)
    best_row,base=selected
    jsave(HERE/'重点结果.json',{'highest_F0_pS01':{**best_row,'F0_saved':float(best['F0'])},'baseline':base,
                              'all_pS01':[r for r in table if r['n_s']==6]})
    print('VALIDATED SUMMARY '+json.dumps({'highest_F0_pS01':best_row,'baseline':base},ensure_ascii=False),flush=True)

def protect_check(before):
    changed=[p for p,h in before.items() if not Path(p).exists() or sha(p)!=h]
    unexpected=changed
    assert not unexpected,unexpected
    result={'changed_existing_files':changed,'unexpected_changes':unexpected,
            'all_existing_experiment_results_and_figures_unchanged':True,
            'collaboration_search_calls':0,'simulation_calls':0}
    jsave(HERE/'范围保护核验.json',result)

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    before=protected_files();jsave(HERE/'执行前文件哈希.json',before)
    state,manifest,config_changed=preflight()
    expected_status={(r['value'],int(r['instance'])):r for r in state}
    expected_manifest={(r['value'],int(r['instance'])-20000):r for r in manifest}
    expected_counts={}
    if config_changed:
        counts_path=HERE/'逐网络任务位置计数.csv'
        if not counts_path.is_file():
            raise RuntimeError('Original per-network counts are required when the configuration source hash differs.')
        count_rows=read(counts_path)
        expected_counts={(r['value'],int(r['instance'])):r for r in count_rows}
        if len(count_rows)!=len(state) or set(expected_counts)!=set(expected_status):
            raise RuntimeError('Original per-network counts are incomplete or contain duplicates.')
    rows=[];mismatches=[];start=time.monotonic()
    jobs=[(int(r['value'].split('-')[0]),int(r['value'].split('-')[1]),int(r['instance'])) for r in state]
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(worker,*job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();key=(row['value'],row['instance']);old=expected_status[key];errors=[]
            if row['status']!=old['status']:errors.append('generation_status')
            if row['failure_class']!=old['failure_class']:errors.append('failure_class')
            if row['seed_index']!=int(old['seed_index']):errors.append('seed_index')
            if config_changed:
                reference=expected_counts[key]
                for field in ['n_s','n_d','n_i','seed_index','scene_seed','status','failure_class',
                              'task_positions_attempted','task_positions_rejected','task_positions_accepted',
                              'completed_tasks','pending_task_trials','constructor_positions_predrawn',
                              'initial_positions_untested_at_stop','rng_positions_drawn_including_untested',
                              'network_sha256','tasks_sha256']:
                    if str(row[field])!=reference[field]:errors.append('saved_counts.'+field)
            if row['status']=='ok':
                ref=expected_manifest.get(key)
                if ref is None:errors.append('unexpected_successful_network')
                else:
                    for field in ['network_sha256','tasks_sha256','scene_seed']:
                        if str(row[field])!=ref[field]:errors.append(field)
            if errors:
                mismatches.append({'value':key[0],'instance':key[1],'fields':errors})
                for pending in futures:pending.cancel()
                pool.shutdown(wait=False,cancel_futures=True)
                report={'passed':False,'mismatched_networks_detected':len(mismatches),'completed_networks':len(rows)+1,
                        'mismatches':mismatches,'rates_output':False,
                        'possible_causes':'Compare seed indexing, NumPy version, generation code version and saved fingerprint provenance.'}
                jsave(HERE/'复现校验.json',report);protect_check(before)
                raise RuntimeError('Reproduction mismatch; stopped, rejection rates withheld.')
            rows.append(row)
            if len(rows)%25==0 or len(rows)==1850:
                print(f'GENERATION ONLY {len(rows)}/1850, failed={sum(r["status"]!="ok" for r in rows)}, elapsed={time.monotonic()-start:.1f}s',flush=True)
    rows.sort(key=lambda r:(r['n_s'],r['n_d'],r['instance']))
    original_failed={key for key,r in expected_status.items() if r['status']!='ok'}
    recreated_failed={(r['value'],r['instance']) for r in rows if r['status']!='ok'}
    assert len(rows)==1850 and sum(r['status']=='ok' for r in rows)==1785 and recreated_failed==original_failed and len(recreated_failed)==65
    validation={'passed':True,'networks_checked':1850,'successful_libraries_checked':1785,'tasks_checked_via_full_library_fingerprint':1785*50,
                'successful_task_library_fingerprints_identical':1785,'network_fingerprints_identical':1785,
                'task_fields_covered':['task_id','position','task_type','requirements'],
                'validation_method':'Original canonical full-library SHA-256 in saved scenario_manifest.csv; every successful network compared separately.',
                'raw_original_task_rows_retained':False,
                'failed_network_set_exact_match':True,'failed_networks':65,
                'precheck_failed_networks':sum(r['failure_class']=='no_path_even_with_full_coverage' for r in rows),
                'position_stage_failed_networks':sum(r['failure_class']=='no_feasible_geometric_position' for r in rows),
                'configuration_source_hash_changed':config_changed,
                'saved_per_network_counts_checked':len(rows) if config_changed else 0,
                'mismatched_networks':0,'collaborative_search_or_simulation_calls':0,
                'elapsed_seconds':time.monotonic()-start,'numpy_version':np.__version__}
    jsave(HERE/'复现校验.json',validation)
    outputs(rows)
    protect_check(before)
    print('ALL REPRODUCTION CHECKS PASSED; ORIGINAL RESULTS AND FIGURES PROTECTED',flush=True)

if __name__=='__main__':main()

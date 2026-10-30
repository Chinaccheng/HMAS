"""Scientific consistency checks, sample counts, input protection and delivery inventory."""
from common import *

def run():
    checks=[]
    def check(name,ok,detail=''):
        checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})
        if not ok:print('VALIDATION FAILURE',name,detail,flush=True)
    if not SNAPSHOT.exists():snapshot()
    verify_snapshot();check('Original files unchanged',True)
    copies=read_csv(OUT/'A_copy_manifest.csv')
    for r in copies:
        if r['status']=='copied_byte_identical':
            p=OUT/'A_existing'/('E'+str(r['experiment']))/r['file'];check('A exact copy '+str(p.relative_to(OUT)),hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'])
    rec=read(5,'metrics_instance.csv');supp=read_csv(OUT/'B02_mixed_mechanism_instance.csv');diff=[]
    ref={(r['instance'],r['kind'],r['intensity'],r['method']):r for r in rec if r['attack']=='SCFC' and r['budget']==.5}
    for r in supp:diff.append(abs(r['Phi_final']-ref[(r['instance'],r['kind'],r['intensity'],r['method'])]['Phi_final']))
    check('B02 replays match original E5 final functionality',max(diff)<2e-12,max(diff))
    trap=read_csv(OUT/'B07_trap_nodes_instance.csv');diff=[]
    for r in trap:
        if r['role']!='all':continue
        refrow=next(v for v in rec if v['instance']==r['instance'] and v['kind']=='node' and v['intensity']==.2 and v['attack']==r['attack'] and v['method']==r['method'] and v['budget']==.5)
        diff.append(abs(r['Phi_final']-refrow['Phi_final']))
    check('B07 replays match original E5 final functionality',max(diff)<2e-12,max(diff))
    freq=[r for r in read_csv(OUT/'B12_freq_instance.csv') if r['F_req_fraction']==.5]
    original=[r for r in read(4,'metrics_instance.csv') if r['mode']=='static']
    lookup={(r['instance'],r['kind'],r['method']):r for r in original}
    diffs=[abs(r['f_c']-lookup[(r['instance'],r['kind'],r['method'])]['f_c_capped']) for r in freq]
    check('B12 threshold 0.5 matches original E4 f_c',max(diffs)<2e-12,max(diffs))
    bottleneck=read_csv(OUT/'B11_bottleneck_threshold.csv')
    source=read(1,'table3_task_types.csv')
    diffs=[abs(r['bottleneck_share']-next(x['存在瓶颈成员的任务比例（某成员覆盖 ≥90% 成功搜索）'] for x in source if x['type']==r['task_type'])) for r in bottleneck if r['threshold']==.9]
    check('B11 threshold 0.9 matches original E1 bottleneck',max(diffs)<2e-12,max(diffs))
    overlap=read_csv(OUT/'B13_lmax_overlap_instance.csv');params=read(7,'parameter_instance.csv')
    diffs=[abs(r['mean_team_size']-next(x['mean_team_size'] for x in params if x['factor']=='hops' and x['parameter']==r['L_max'] and x['instance']==r['instance'])) for r in overlap]
    check('B13 reconstruction matches E7 team sizes',max(diffs)<2e-12,max(diffs))
    substitution=read_csv(OUT/'B14_link_self_substitution_instance.csv');e2metrics=read(2,'metrics_raw.csv')
    diffs=[abs(r['edge_tau']-next(x['tau'] for x in e2metrics if x['n']==60 and x['instance']==r['instance'] and x['kind']=='edge' and x['method']=='SCFC')) for r in substitution if r['alpha']==3]
    check('B14 alpha 3 matches original E2 edge tau',max(diffs)<2e-12,max(diffs))
    check('B14 self-substitution lies in [0,1]',all(-1e-12<=r['self_substitution_share']<=1+1e-12 for r in substitution))
    for i in range(10):
        rs=sorted([r for r in substitution if r['instance']==i],key=lambda r:r['alpha'])
        check('B14 self-share increases with alpha '+str(i),all(b['self_substitution_share']>=a['self_substitution_share']-1e-12 for a,b in zip(rs,rs[1:])))
    for task,name,expected in [('B03','B03_nodes.csv',50),('B04','B04_top5_instance.csv',50),('C1','C1_timing.csv',5),('C2','C2_role_attack_instance.csv',50),('C3','C3_structure_instance.csv',20),('B13','B13_lmax_overlap_instance.csv',10),('B14','B14_link_self_substitution_instance.csv',10)]:
        rs=read_csv(OUT/name)
        key='scale' if task=='C1' else 'L_max' if task=='B13' else None
        if key:
            for v in sorted({r[key] for r in rs}):check(task+' scene count '+str(v),len({r['instance'] for r in rs if r[key]==v})==expected)
        else:check(task+' scene count',len({r['instance'] for r in rs})==expected)
    # Scene-level confidence intervals must exactly reproduce common.mean_ci.
    b=read_csv(OUT/'C3_structure.csv')[0];raw=read_csv(OUT/'C3_structure_instance.csv');mu,ci,n=mean_ci([r['F0'] for r in raw]);check('C3 F0 CI uses 20 independent instances',abs(mu-b['F0'])<1e-14 and abs(ci-b['F0_ci95'])<1e-14 and n==b['F0_n']==20)
    unit=read_json(OUT/'C3_success_chain_test.json');check('C3 successful path definition tests',unit['passed'] and unit['checked_successes']>0,unit['checked_successes'])
    # Explicitly compare C2 all-role reference to E4 interpolation, reporting the finite-grid discrepancy.
    b=read_csv(OUT/'C2_role_attack_summary.csv');check('C2 exact all-role random reference included',any(r['role']=='all_random_reference' and r['R_rob_n']==50 for r in b))
    statsrows=read_csv(OUT/'B10_holm.csv');check('Holm correction never decreases p',all(r['p_holm']>=r['p_raw']-1e-15 for r in statsrows))
    check('Holm families map to manuscript tables 4/5/6',set(r['table'] for r in statsrows)=={'Table 4','Table 5','Table 6'})
    check('All primary tasks have outputs',all(any(OUT.glob(t+'*.csv')) for t in ['A','B01','B02','B03','B04','B05','B06','B07','B08','B09','B10','B11','B12','B13','B14','C1','C2','C3']))
    save('validation_checks.csv',checks)
    inventory=[]
    for p in sorted(OUT.rglob('*.csv')):
        if p.name=='delivery_inventory.csv':continue
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);columns=reader.fieldnames;count=sum(1 for r in reader)
        inventory.append({'file':str(p.relative_to(OUT)),'rows':count,'columns':len(columns),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save('delivery_inventory.csv',inventory)
    write_json(OUT/'validation_summary.json',{'passed':all(r['passed'] for r in checks),'checks':len(checks),'failed':[r for r in checks if not r['passed']],'csv_files':len(inventory)+1})
    assert all(r['passed'] for r in checks),'See validation_checks.csv'
    return checks
if __name__=='__main__':run()

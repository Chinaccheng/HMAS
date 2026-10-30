"""v11统一入口：全部实验共用experiment_config；基准K300、λ10、Lmax12及可行任务拒绝采样。"""
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--experiments',type=int,nargs='+',choices=range(1,8),default=list(range(1,8)))
    p.add_argument('--instances',type=int,default=50)
    p.add_argument('--pilot-instances',type=int,default=10)
    p.add_argument('--K',type=int,default=None)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--scale-instances',type=int,default=20)
    p.add_argument('--truth-K',type=int,default=1000)
    p.add_argument('--random-repeats',type=int,default=20)
    p.add_argument('--force',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--compute-only',action='store_true')
    args=p.parse_args()
    for number in [n for n in [1,2,6,3,4,5,7] if n in args.experiments]:
        keys=['instances','K','workers']+(['pilot_instances'] if number==1 else ['scale_instances','truth_K','random_repeats'] if number==2 else [] if number==3 else ['random_repeats'])
        values={k:getattr(args,k) for k in keys};values['K']=args.K if args.K is not None else 300
        opts=[arg for k in keys for arg in ['--'+k.replace('_','-'),str(values[k])]]
        if args.force:opts.append('--force')
        if args.plot_only:opts.append('--plot-only')
        if args.compute_only:opts.append('--compute-only')
        folder={1:'E1_基准场景',2:'E2_识别准确性',3:'E3_消融机理',4:'E4_定向打击',5:'E5_资源恢复',6:'E6_结构因素',7:'E7_参数敏感性'}[number]
        subprocess.run([sys.executable,str(ROOT/folder/'run.py'),*opts],check=True,cwd=ROOT.parent)
    print('所选实验运行完成。')

if __name__=='__main__':main()

"""Reproduce in the required P1 -> P2 -> P3 order, serially."""
from common import *
TASKS=['A_existing_exports', 'B01_link_mixed_recovery', 'B02_mixed_mechanism', 'B03_D_bias', 'B04_top5_nonadditivity', 'B05_figure_values', 'B06_PhiP_eta', 'B07_trap_nodes', 'B08_structure_scale_parameters', 'B09_small_statistics', 'B10_holm', 'C1_timing', 'B11_bottleneck_threshold', 'B12_freq_sensitivity', 'C2_role_attack', 'C3_relay_experiment', 'B13_lmax_overlap', 'B14_link_self_substitution']
if __name__=='__main__':
    # Record the inputs for this run, not a snapshot from an earlier run.
    snapshot()
    verify_snapshot();environment()
    subprocess.run([sys.executable,'-B',str(HERE/'build_relay_copy.py')],check=True)
    for task in TASKS:
        print('RUN',task,flush=True)
        subprocess.run([sys.executable,'-B','-u',str(HERE/(task+'.py'))],check=True,cwd=HERE)
    subprocess.run([sys.executable,'-B',str(HERE/'validate_results.py')],check=True,cwd=HERE)

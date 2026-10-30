"""Generate the local generalized kernel copy without touching original model.py."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
src=(ROOT/'model.py').read_text()
src=src.replace('from .config import Config','from UAV.config import Config').replace('from config import Config','from UAV.config import Config')
src=src.replace('from collections import Counter','from collections import Counter\nfrom relay_definition import ALLOW_MATRIX, BIT_MATRIX, STATE_COUNT, ROLE_CODES, TRANSITIONS, RELATION_BITS as RELAY_BITS, MINIMUM_HOPS, NONTERMINAL_ROLES, UNRESTRICTED_MASK')
src=src.replace('N, 8','N, STATE_COUNT').replace('N * 8','N * STATE_COUNT').replace('range(8)','range(STATE_COUNT)')
src=src.replace('((a == 0 and b in (0, 1)) or (a == 1 and b in (0, 1, 2)))','ALLOW_MATRIX[a, b]')
src=src.replace('starts_mask[k, target] or b == 1 or','starts_mask[k, target] or UNRESTRICTED_MASK[b] or')
src=src.replace('1 if a == 0 and b == 0 else (2 if a == 1 and b == 1 else (4 if a == 1 and b == 0 else 0))','BIT_MATRIX[a, b]')
src=src.replace('{"S": 0, "D": 1, "I": 2}[r]','ROLE_CODES[r]')
src=src.replace('if roles[i] == "D"','if roles[i] in NONTERMINAL_ROLES - {"S"}')
src=src.replace('if self.roles[i] == "D"','if self.roles[i] in NONTERMINAL_ROLES - {"S"}')
src=src.replace('return 2 + bool(self.requirements & 1) + bool(self.requirements & 2) + 2 * bool(self.requirements & 4)','return MINIMUM_HOPS[self.requirements]')
src=re.sub(r'^ALLOWED = .*$', 'ALLOWED = TRANSITIONS',src,flags=re.M)
src=re.sub(r'^RELATION_BITS = .*$', 'RELATION_BITS = RELAY_BITS',src,flags=re.M)
src=re.sub(r'^TASK_NAMES = .*$', 'from relay_definition import TASK_NAMES',src,flags=re.M)
# SCFC overlap remains the original formula but the contributing roles are explicit parameters.
src=src.replace('if self.roles[j] != "I"','if self.roles[j] in NONTERMINAL_ROLES')
(HERE/'relay_kernel.py').write_text(src)
# Generalize exactly the existing simple-path enumerator for the new state table.
e2=(ROOT/'E2_识别准确性/run.py').read_text();start=e2.index('@njit(cache=False)');end=e2.index('\ndef cmlc(',start)
s=e2[start:end].replace('n,8','n,STATE_COUNT').replace('range(8)','range(STATE_COUNT)')
s=s.replace('((a==0 and (b==0 or b==1)) or (a==1 and (b==0 or b==1 or b==2)))','ALLOW_MATRIX[a,b]')
s=s.replace('starts[j] or b==1 or','starts[j] or UNRESTRICTED_MASK[b] or')
s=s.replace('roles[i]==2','roles[i]==TERMINAL_CODE').replace('b==2','b==TERMINAL_CODE')
s=s.replace('1 if a==0 and b==0 else (2 if a==1 and b==1 else (4 if a==1 and b==0 else 0))','BIT_MATRIX[a,b]')
s=s.replace("{'S':0,'D':1,'I':2}[r]",'ROLE_CODES[r]')
(HERE/'relay_loops.py').write_text('import numpy as np\nfrom numba import njit\nfrom relay_definition import *\n'+s)
print('Generated local Relay kernels')

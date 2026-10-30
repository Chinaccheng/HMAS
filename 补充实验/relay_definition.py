"""C3 definition revised by the user on 2026-10-27; no explicit D-visit acceptance rule."""
from collections import deque
import numpy as np
ROLE_CODES={'S':0,'D':1,'I':2,'R':3}
TRANSITIONS={('S','S'),('S','D'),('D','S'),('D','D'),('D','I'),('D','R'),('R','R'),('R','I')}
RELATION_BITS={('S','S'):1,('D','D'):2,('D','S'):4,('D','R'):8}
TASK_NAMES=('基础打击','协同侦察','协同决策','反馈确认','远程打击')
REQUIREMENTS=(0,1,2,4,8)
NONTERMINAL_ROLES=frozenset({'S','D','R'})
TERMINAL_CODE=ROLE_CODES['I']
UNRESTRICTED_MASK=np.zeros(len(ROLE_CODES),np.bool_)
for role in NONTERMINAL_ROLES-{'S'}:UNRESTRICTED_MASK[ROLE_CODES[role]]=True
ALLOW_MATRIX=np.zeros((len(ROLE_CODES),len(ROLE_CODES)),np.bool_);BIT_MATRIX=np.zeros_like(ALLOW_MATRIX,dtype=np.int64)
for a,b in TRANSITIONS:ALLOW_MATRIX[ROLE_CODES[a],ROLE_CODES[b]]=True
for (a,b),bit in RELATION_BITS.items():BIT_MATRIX[ROLE_CODES[a],ROLE_CODES[b]]=bit
STATE_COUNT=1<<len(RELATION_BITS)

def minimum_chain(requirement):
    q=deque([('S',requirement,('S',))]);seen={('S',requirement)}
    while q:
        current,rem,path=q.popleft()
        for a,b in sorted(TRANSITIONS):
            if a!=current or (b=='I' and rem):continue
            rest=rem & ~RELATION_BITS.get((a,b),0)
            newpath=path+(b,)
            if b=='I' and rest==0:return newpath
            if (b,rest) not in seen:seen.add((b,rest));q.append((b,rest,newpath))
    raise ValueError('Unrealizable task requirement')
MINIMUM_HOPS={req:len(minimum_chain(req))-1 for req in REQUIREMENTS}

"""Definition-1 tests must pass before any C3 production experiment."""
import unittest
from relay_model import *

class RelayTests(unittest.TestCase):
    def test_minimum_hops_and_D_guarantee(self):
        self.assertEqual([MINIMUM_HOPS[q] for q in REQUIREMENTS],[2,3,3,4,3])
        # Exhaust all role/rem states before the first D to certify the transition table property.
        frontier={'S'};seen=set()
        while frontier:
            a=frontier.pop();seen.add(a)
            for x,y in TRANSITIONS:
                if x==a and y!='D':
                    self.assertNotEqual(y,'I')
                    if y not in seen:frontier.add(y)
        for q in REQUIREMENTS:self.assertIn('D',minimum_chain(q)[:-1])
    def test_all_success_chains(self):
        checked=Counter();checked_total=0
        for i in [0,1]:
            m=RelayModel(i);d=draws_for(m,i,300)
            def check(k,r,path):
                nonlocal checked_total
                roles=[m.roles[n] for n in path];t=m.tasks[k]
                self.assertIn(path[0],m.coverage_s[k]);self.assertEqual(roles[0],'S')
                self.assertEqual(roles[-1],'I');self.assertEqual(roles.count('I'),1)
                self.assertIn(path[-1],m.coverage_i[k]);self.assertIn('D',roles[:-1])
                self.assertLessEqual(len(path)-1,m.config.max_hops)
                relations=list(zip(roles,roles[1:]));self.assertTrue(all(x in TRANSITIONS for x in relations))
                mask=0
                for relation in relations:mask|=RELATION_BITS.get(relation,0)
                self.assertEqual(mask&t.requirements,t.requirements)
                self.assertGreaterEqual(len(path)-1,t.minimum_hops)
                checked[t.task_type]+=1;checked_total+=1
            b=m.evaluate(random_inputs=d,on_success=check)
            ref=m.evaluate_reference(random_inputs=d,collect=False)
            self.assertAlmostEqual(b.F,ref.F,places=12)
            self.assertTrue(all(checked[t]>0 for t in range(5)))
        write_json(OUT/'C3_success_chain_test.json',{'passed':True,'checked_successes':checked_total,'by_task_type':dict(checked),'minimum_hops':MINIMUM_HOPS,'minimum_chains':{q:minimum_chain(q) for q in REQUIREMENTS}})
    def test_batch_and_scalar(self):
        m=RelayModel(0);d=draws_for(m,0,300)
        states=[NetworkState({n},set()) for n in [0,20,59]]+[NetworkState(set(),{m.initial_edges[0]})]
        bat=m.evaluate_functions(states,random_inputs=d)
        single=[m.evaluate(s,random_inputs=d,collect=False).F for s in states]
        np.testing.assert_allclose(bat,single,rtol=0,atol=2e-14)
    def test_no_legacy_change_and_role_counts(self):
        m=RelayModel(0);self.assertEqual(dict(Counter(m.roles)),{'S':20,'D':10,'R':10,'I':20})
        self.assertNotIn(('S','R'),TRANSITIONS);self.assertNotIn(('R','D'),TRANSITIONS)
        self.assertTrue(all(b!='I' for a,b in RELATION_BITS))

if __name__=='__main__':unittest.main(verbosity=2)

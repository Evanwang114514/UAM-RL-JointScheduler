"""只检查纯逻辑与封存源码，不导入Torch、不加载模型、不运行交通仿真。"""
import ast
import base64
import gzip
import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('uam_mechanism_unit', ROOT/'diagnose_uam_mechanisms_1h_ONEFILE.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class PureLogicTests(unittest.TestCase):
    def test_payload_exact_and_parseable(self):
        raw = gzip.decompress(base64.b64decode(D.ADAPTER_GZIP_B64))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), D.ADAPTER_SHA)
        # Git跨平台检出可转换换行；封存哈希仍检查原始字节，源码对照按统一换行核验。
        self.assertEqual(raw.replace(b'\r\n',b'\n'),
                         (ROOT/'diagnose_uam_decision_semantics_notrain_ONEFILE.py').read_bytes().replace(b'\r\n',b'\n'))
        ast.parse(raw.decode('utf-8-sig'))

    def test_sampling_passes_partition_the_same_thirty_slots(self):
        for control in ('J3', 'LQ'):
            values = [D.pass_slots(control, i) for i in range(3)]
            self.assertEqual([len(v) for v in values], [6,12,12])
            merged = [x for v in values for x in v]
            self.assertEqual(len(set(merged)), 30)
            self.assertEqual(sorted(merged, key=lambda x:x[1]), D.normal_slots(control))
        self.assertEqual(sum(k=='air_only' for k,i in D.normal_slots('J3')),10)
        self.assertEqual(sum(k=='air_only' for k,i in D.normal_slots('LQ')),0)

    def test_four_arm_interaction_needs_actual_matching_opportunities(self):
        costs={'00':100,'10':105,'01':102,'11':70}
        x=D.causal_four(costs,True,10)
        self.assertTrue(x['pair_required'])
        self.assertEqual(x['pair_extra'],30)
        self.assertEqual(x['interaction'],-37)
        self.assertFalse(D.causal_four(costs,False,10)['pair_required'])
        self.assertFalse(D.causal_four({'00':100,'10':60,'01':90,'11':70},True,10)['pair_required'])

    def test_mean_first_not_clairvoyant_minimum(self):
        self.assertEqual(D.choose_mean([[0,100],[45,45]]),1)
        self.assertEqual(D.choose_mean([[3,7],[4,6]],reference=1),1)

    def test_branch_quality_preserves_signed_cost_and_censoring(self):
        def row(cost, p, air=None):
            plan=dict(p=p)
            if air is not None:plan['air']=air
            return dict(plan=plan,total=cost,operating=cost,tail=0,nonabsorbing=cost,
                phases=dict(ground=10,wait=cost-10,flight=0,other=0),
                cohorts=dict(focal=20,existing=cost-20,future=0),
                points={str(h):dict(total=cost) for h in (5,30,60,120)},
                first_post=str((p,air)),first_air_post=str(air),remaining=2)
        values=[row(100,0),row(80,1),row(95,0,1)]
        q=D.quality_record(values,dict(passenger=0),10)
        self.assertEqual(q['passenger_only_gain'],20)
        self.assertEqual(q['aircraft_only_gain'],5)
        self.assertEqual(q['reference_remaining'],2)
        self.assertEqual(sum(q['phase_gain'].values()),q['gain'])

    def test_delay_censor_is_not_zero_and_proxies_are_not_oracle(self):
        rows=[]
        for p,cost in [(0,100),(1,80)]:
            rows.append(dict(plan=dict(p=p),total=cost,focal='a',start=10,
                timeline={'a':dict(queue_seen=None,board_seen=35 if p else None)}))
        scores=[dict(access=5,current_score=1,ready_score=5),dict(access=15,current_score=2,ready_score=3)]
        x=D.delay_record(rows,scores,dict(passenger=0),10)
        self.assertTrue(x['proxies_disagree'])
        self.assertEqual(x['policy_loss'],20)
        self.assertEqual(x['ready_proxy_loss'],0)
        self.assertIsNone(x['choices'][0]['queue_delay'])
        self.assertTrue(x['choices'][0]['censored_board'])

    def test_expanded_plan_has_no_performance_based_selection(self):
        from types import SimpleNamespace
        cases=[]
        for seed in (1801,1802,1803,1804):
            for control in ('J3','LQ'):
                for network in ('LSTM','TRANSFORMER'):
                    for gamma in (.99,1.):
                        cases.append(dict(case=f'{seed}_{control}_{network}_{gamma}',controller='MODEL',
                            training_seed=seed,task=dict(seed=seed,control=control,network=network,gamma=gamma)))
        fake=SimpleNamespace(locate_cases=lambda source:(cases,{}),locate_bank=lambda source,plan:['4303','4304'])
        chosen,bank=D.cases_and_bank(fake,Path('.'))
        self.assertEqual(len(chosen),18)
        self.assertEqual(sum(c['controller']=='READY' for c in chosen),2)
        self.assertEqual(len({c['training_seed'] for c in chosen if c['controller']=='MODEL'}),4)
        self.assertTrue(all(c['task']['gamma']==.99 for c in chosen))
        self.assertEqual(D.PLANNED_JOBS,36+32+10+36+36)

    def test_delay_costs_use_same_absolute_boundary(self):
        rows=[]
        for p,cost,queue,early in [(0,100,20,0),(1,80,25,10)]:
            rows.append(dict(plan=dict(p=p),total=cost,focal='a',start=10,
                cost_curve=[0]+[early]*15+[cost],
                timeline={'a':dict(queue_seen=queue,board_seen=30)}))
        scores=[dict(access=5,current_score=1,ready_score=5),dict(access=15,current_score=2,ready_score=3)]
        x=D.delay_record(rows,scores,dict(passenger=0),10)['delayed_effect_comparison']
        self.assertEqual(x['common_minutes'],15)
        self.assertEqual(x['loss_before_common_entry_bound'],-10)
        self.assertEqual(x['loss_after_common_entry_bound'],30)


if __name__=='__main__':unittest.main()

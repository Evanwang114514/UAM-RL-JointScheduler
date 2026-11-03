"""只检验纯逻辑和源码封存，不导入学习库、不启动仿真或训练。"""
import ast
import base64
import hashlib
import io
import math
import runpy
import tarfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain=runpy.run_path(str(ROOT/'train_uam_space_time_14h_ONEFILE.py'),run_name='pure_test_plain')
        cls.hot=runpy.run_path(str(ROOT/'train_uam_hot_transfer_40h_ONEFILE.py'),run_name='pure_test_hot')
        cls.diag=runpy.run_path(str(ROOT/'diagnose_uam_threeway_comprehensive_ONEFILE.py'),run_name='pure_test_diag')

    def test_matrix_and_budget(self):
        tasks=self.plain['ordinary_tasks']()
        self.assertEqual(len(tasks),16)
        self.assertEqual(len({t['id'] for t in tasks}),16)
        self.assertEqual(16*self.plain['PHYSICAL_STEPS'],48_168_960)
        for seed in self.plain['SEEDS']:
            self.assertEqual({(t['space'],t['temporal']) for t in tasks if t['seed']==seed},
                             {(False,False),(True,False),(False,True),(True,True)})

    def test_hot_is_two_methods_with_controls(self):
        tasks=self.hot['hot_tasks']()
        self.assertEqual(len(tasks),16)
        self.assertEqual({t['arm'] for t in tasks},{'PPO_TF','DV3_H15MIN','HAPPO_NATIVE','MAPPO_MATCHED'})
        self.assertTrue(all(not t['temporal'] for t in tasks))
        self.assertEqual(16*self.hot['HOT_STEPS'],16_056_320)
        self.assertEqual(4*sum(self.hot['HOT_HOURS'].values()),40.)

    def test_discount_conversion(self):
        self.assertAlmostEqual(math.sqrt(.99)**2,.99)
        self.assertAlmostEqual(math.sqrt(.95)**2,.95)
        horizon=1/(1-math.sqrt(.99))
        self.assertAlmostEqual((1-1/horizon)**2,.99)

    def test_native_marl_is_macro_not_alternating_noop(self):
        code=(ROOT/'train_uam_hot_transfer_40h_ONEFILE.py').read_text(encoding='utf-8')
        self.assertIn("buffers[0].obs[k]=obs",code)
        self.assertIn("buffers[1].obs[k]=mid",code)
        self.assertIn('author_train(runner)',code)
        self.assertIn('value_norm_statistics',code)

    def test_new_suite_collection_names_agree(self):
        text=(ROOT/'diagnose_uam_threeway_comprehensive_ONEFILE.py').read_text(encoding='utf-8')
        self.assertIn('uam_hot_transfer_16run_40h_v1',text)
        self.assertIn('uam_space_time_16run_48m_v1',text)
        self.assertNotIn('uam_dreamerv3_12run_40h_v1',text)

    def test_canonical_att_is_not_silently_replaced_with_integral(self):
        source=(ROOT/'train_uam_space_time_14h_ONEFILE.py').read_text(encoding='utf-8')
        self.assertIn('full_att=sum(durations)/generated',source)
        self.assertIn('completed_episode_integral_cost=',source)

    def test_gate_delays_are_physical_ticks(self):
        for delay in (0,1,3):
            calls=[]
            scenario=SimpleNamespace(time=0,_all_evtols={})
            env=SimpleNamespace(scenario=scenario,_priority=1)
            env._dispatch_hook=lambda s:calls.append(s.time)
            gate=self.plain['DispatchGate'](env,SimpleNamespace())
            gate.request=(1,delay)
            for tick in range(delay+1):
                scenario.time=tick
                scenario._strict_dispatch_returns()
            self.assertEqual(calls,[delay])
            self.assertEqual(gate.remaining,0)
            self.assertFalse(gate.release)

    def test_person_gain_conservation(self):
        identity={'a':{'born':0},'b':{'born':1}}
        a=dict(individual={'a':10,'b':20},request_identity=identity)
        b=dict(individual={'a':15,'b':10},request_identity=identity)
        result=self.plain['compare_people'](a,b)
        self.assertEqual(result['total_gain'],5)
        self.assertEqual(result['benefited'],1)
        self.assertEqual(result['harmed'],1)
        self.assertEqual(result['worst_harm'],5)
        self.assertEqual(result['gained_person_minutes']-result['lost_person_minutes'],result['total_gain'])

    def test_unpaired_people_fail_closed(self):
        value=self.plain['compare_people'](dict(individual={},request_identity={}),dict(individual={},request_identity={'x':1}))
        self.assertEqual(value['status'],'UNPAIRED_REQUEST_IDENTITIES')

    def test_frozen_sources_compile_and_match_hash(self):
        for module,blob_key,sha_key in [(self.hot,'RUNTIME_B85','RUNTIME_SHA'),
            (self.diag,'DIAG_B85','DIAG_SHA'),(self.diag,'STAGE_B85','STAGE_SHA')]:
            raw=zlib.decompress(base64.b85decode(module[blob_key]))
            self.assertEqual(hashlib.sha256(raw).hexdigest(),module[sha_key])
            ast.parse(raw.decode('utf-8-sig'))

    def test_author_bundle_is_safe_and_licensed(self):
        raw=zlib.decompress(base64.b85decode(self.hot['VENDOR_B85']))
        self.assertEqual(hashlib.sha256(raw).hexdigest(),self.hot['VENDOR_SHA'])
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
            names=[]
            for member in tf:
                self.assertTrue(member.isfile())
                self.assertFalse(Path(member.name).is_absolute())
                self.assertNotIn('..',Path(member.name).parts)
                self.assertEqual(hashlib.sha256(tf.extractfile(member).read()).hexdigest(),self.hot['VENDOR_FILE_HASHES'][member.name])
                names.append(member.name)
            self.assertIn('LICENSE',names)
            self.assertIn('dreamerv3/agent.py',names)
            self.assertIn('embodied/jax/agent.py',names)

    def test_sampling_is_not_outcome_selected(self):
        contexts=[dict(time=t,meaningful=True,unfinished=30,active=t%2==0) for t in range(20,400)]
        sampled=self.diag['sample_contexts'](contexts,12)
        self.assertLessEqual(len(sampled),12)
        self.assertEqual(sampled,self.diag['sample_contexts'](contexts,12))
        self.assertTrue(any(c['time']>=300 for c in sampled))

if __name__=='__main__':unittest.main()

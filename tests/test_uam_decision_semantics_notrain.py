"""只验证纯逻辑和封存内容；不导入学习库、不加载模型、不运行交通回放。"""
import ast
import base64
import hashlib
import importlib.util
import json
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('decision_semantics_unit', ROOT / 'diagnose_uam_decision_semantics_notrain_ONEFILE.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DiagnosticLogicTests(unittest.TestCase):
    def test_sealed_runtime_has_exact_latest_training_source(self):
        raw = zlib.decompress(base64.b85decode(d.FROZEN_RUNTIME_B85))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), d.SOURCE_SCRIPT_SHA)
        self.assertEqual(raw, (ROOT / 'train_uam_credit_horizon_18h_ONEFILE.py').read_bytes())
        ast.parse(raw.decode('utf-8-sig'))
        base = zlib.decompress(base64.b85decode(d.FROZEN_BASE_B85))
        self.assertEqual(hashlib.sha256(base).hexdigest(), d.FROZEN_BASE_SHA)
        self.assertIn('def clone_env', base.decode())

    def test_forecast_can_see_only_revealed_history(self):
        people = {str(i): NS(spawn_time=i, origin_position=[i, 0], destination_position=[0, i]) for i in range(30)}
        class NoFuture:
            persons = people
            @property
            def file_data(self):
                raise AssertionError('合法预测偷读了未来文件')
        env = NS(scenario=NS(time=30, passenger_generation_end_time=45, persons=NoFuture()))
        history = d.public_history(env)
        first = d.forecast_world(history, 42, 120)
        self.assertEqual(first, d.forecast_world(history, 42, 120))
        known = {(tuple(p.origin_position), tuple(p.destination_position)) for p in people.values()}
        self.assertTrue(first)
        for t, rows in first.items():
            self.assertGreaterEqual(t, 30); self.assertLessEqual(t, 45)
            for row in rows:
                self.assertIn((tuple(row['origin']), tuple(row['destination'])), known)

    def test_forecast_does_not_erase_future_when_history_insufficient(self):
        self.assertIsNone(d.forecast_world(dict(now=0, generation_end=300, revealed=[]), 2, 120))

    def test_tail_label_requires_all_declared_physical_conditions(self):
        planes = {str(i): NS(state=NS(name='IDLE'), passenger_ids=[]) for i in range(12)}
        stations = {'0': NS(person_list=['a', 'b', 'c']), '1': NS(person_list=['d']), '2': NS(person_list=[])}
        s = NS(time=400, passenger_generation_end_time=300, waiting_decisions=[],
               persons=NS(persons={x: None for x in 'abcd'}), finished_ids=[], _all_evtols=planes,
               vertiports=NS(vertiport_list=stations))
        env = NS(scenario=s)
        self.assertTrue(d.tail_absorbing(env))
        planes['0'].state.name = 'CHARGING'; self.assertFalse(d.tail_absorbing(env))
        planes['0'].state.name = 'IDLE'
        stations['0'].person_list.append('d'); stations['1'].person_list.clear()
        self.assertFalse(d.tail_absorbing(env))

    def test_selection_is_minimum_of_mean_not_mean_of_clairvoyant_minima(self):
        actions = [[0, -1], [1, -1]]
        self.assertEqual(d.choose([[0, 100], [45, 45]], actions, actions[0]), 1)
        self.assertEqual(d.choose([[3, 7], [4, 6]], actions, actions[1]), 1)

    def test_fixed_cohorts_and_occupancy_conserve_person_time(self):
        persons = {'a': NS(sub_state='to_vertiport'), 'b': NS(sub_state='wait_uam'), 'c': NS(sub_state='fly')}
        env = NS(scenario=NS(persons=NS(persons=persons), finished_ids=[]))
        n, cohorts, phases = d.cost_tick(env, {'a', 'b'}, 'a')
        self.assertEqual(n, 3); self.assertEqual(cohorts, dict(focal=1, existing=1, future=1))
        self.assertEqual(sum(phases.values()), n)

    def test_selection_loss_and_signed_external_costs(self):
        def row(total, focal, existing, future):
            points = {str(h): dict(total=total) for h in (1, 5, 15, 30, 60, 120, 240)}
            return dict(total=total, cohorts=dict(focal=focal, existing=existing, future=future),
                        components=dict(ground=0, wait=total, flight=0, other=0), points=points, remaining=0)
        actions = [[0, 0], [1, 0], [0, 1], [1, 1]]
        values = [row(100, 10, 40, 50), row(85, 12, 33, 40), row(90, 10, 35, 45), row(80, 12, 28, 40)]
        result = d.compare_rows(values, actions, actions[0], 2)
        self.assertEqual(result['avoidable_person_minutes'], 20)
        self.assertEqual(result['focal_gain'], -2)
        self.assertEqual(result['existing_gain'] + result['future_gain'], 22)
        self.assertEqual(result['joint_extra_beyond_best_single'], 5)
        self.assertFalse(result['near_best'])

    def test_sampling_is_fixed_normal_stage_not_selected_failure(self):
        indices = d.selected_indices('order_4303')
        self.assertEqual(indices, d.selected_indices('order_4303'))
        self.assertEqual(len(set(indices)), 4)
        for value, bounds in zip(indices, ((25, 75), (100, 150), (175, 225), (225, 240))):
            self.assertTrue(bounds[0] <= value < bounds[1])


    def test_thirty_minute_selection_is_balanced_not_performance_selected(self):
        models = [dict(controller='MODEL', task=dict(gamma=g, seed=s, control=c, network=n))
                  for g in (.99, 1.) for s in (1801, 1802, 1803, 1804)
                  for c in ('J3', 'LQ') for n in ('LSTM', 'TRANSFORMER')]
        ready = dict(controller='READY', task=dict(gamma=.99, seed=1801, control='J3', network='TRANSFORMER'))
        self.assertEqual(len(d.phase_cases([ready] + models, 'baseline')), 17)
        deep = d.phase_cases([ready] + models, 'deep')
        self.assertEqual(len(deep), 9)
        for seed in (1801, 1802, 1803, 1804):
            self.assertEqual({c['task']['control'] for c in deep if c['controller'] == 'MODEL' and c['task']['seed'] == seed}, {'J3', 'LQ'})


if __name__ == '__main__': unittest.main()

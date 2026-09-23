"""Public-API contract and strategy checks; no hidden environment inspection."""
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
from agent import Agent
from scoring_core import CHANNELS, validate_strategy


class PublicEnv:
    def __init__(self, budget=100000, contacts=15000, effect=0.2, failure=None):
        self.channels = CHANNELS.copy()
        self.tariffs = pd.DataFrame({'tariff_plan_code': ['a', 'b', 'c']})
        self.customer_profile = pd.DataFrame([
            {'ID_NUMBER': i, 'current_tariff': 'a' if i < 500 else 'b',
             'arpu_segment': 'MID', 'predicted_arpu': 2000.}
            for i in range(1000)])
        self.remaining_budget, self.remaining_contacts = budget, contacts
        self.pilots_left, self.pilot_history = 20, []
        self.effect, self.failure, self.calls = effect, failure, 0

    def run_pilot(self, target_tariff, channel, n_customers, **filters):
        self.calls += 1
        assert 10 <= n_customers <= 200
        cost = n_customers * self.channels[channel]['cost_per_contact']
        assert cost <= self.remaining_budget
        assert n_customers <= self.remaining_contacts and self.pilots_left > 0
        if self.failure == 'first' and self.calls == 1:
            raise RuntimeError('Transient API failure')
        if self.failure == 'all':
            raise OSError('API unavailable')
        self.remaining_budget -= cost
        self.remaining_contacts -= n_customers
        self.pilots_left -= 1
        result = {'n_customers': n_customers,
                  'observed_lift_ratio': self.effect * self.channels[channel]['conversion_multiplier']}
        self.pilot_history.append(result)
        return result


def historical_prior():
    return pd.DataFrame([{'tariff_plan_code_from': cur, 'tariff_plan_code_to': target,
                          'arpu_segment': 'MID', 'ratio': .3, 'n': 100, 'std': .5}
                         for cur in ['a', 'b'] for target in ['a', 'b', 'c'] if cur != target])


class AgentTests(unittest.TestCase):
    def run_agent(self, env):
        agent = Agent()
        with patch.object(agent, '_build_prior', return_value=historical_prior()):
            campaigns = agent.act(env)
        if campaigns:
            validate_strategy(pd.DataFrame(campaigns), env.tariffs)
        self.assertLessEqual(len(campaigns), 10)
        self.assertGreaterEqual(env.remaining_budget, 0)
        self.assertGreaterEqual(env.remaining_contacts, 0)
        self.assertGreaterEqual(env.pilots_left, 0)
        cells = [(c['filter_current_tariff'], c['filter_arpu_segment']) for c in campaigns]
        self.assertEqual(len(cells), len(set(cells)))
        return agent, campaigns

    def test_small_budget_and_contact_reservation(self):
        env = PublicEnv(budget=100, contacts=61)
        _, campaigns = self.run_agent(env)
        self.assertTrue(env.pilot_history)
        self.assertTrue(campaigns)
        self.assertGreaterEqual(env.remaining_contacts, 1)

    def test_failure_does_not_prevent_next_experiment(self):
        with warnings.catch_warnings(record=True) as logged:
            env = PublicEnv(failure='first')
            _, campaigns = self.run_agent(env)
        self.assertTrue(logged)
        self.assertTrue(campaigns)
        self.assertGreater(len(env.pilot_history), 0)

    def test_all_failures_produce_cheap_fallback(self):
        with warnings.catch_warnings(record=True):
            _, campaigns = self.run_agent(PublicEnv(failure='all'))
        self.assertEqual(len(campaigns), 1)
        self.assertEqual(campaigns[0]['channel'], 'push')

    def test_pilots_override_positive_history(self):
        positive, good = self.run_agent(PublicEnv(effect=.5))
        negative, bad = self.run_agent(PublicEnv(effect=-.8))
        self.assertTrue(good)
        self.assertEqual(bad, [])  # no positive-expectation feasible campaign
        self.assertNotEqual(positive.diagnostics, negative.diagnostics)

    def test_deterministic_and_disjoint(self):
        a, first = self.run_agent(PublicEnv())
        b, second = self.run_agent(PublicEnv())
        self.assertEqual(first, second)
        self.assertEqual(a.pilot_log, b.pilot_log)

    def test_prior_reliability_and_call_saturation(self):
        agent = Agent()
        agent.cells = {('a', 'MID'): np.array([0., 2000.])}
        prior = historical_prior()
        prior.loc[prior.tariff_plan_code_to == 'b', 'n'] = 5
        prior.loc[prior.tariff_plan_code_to == 'c', 'n'] = 500
        candidates = agent._rank_candidates(prior, {'a', 'b', 'c'})
        self.assertLess(candidates[0]['prior_precision'], candidates[1]['prior_precision'])
        self.assertLessEqual(candidates[1]['prior_precision'], 20 / .804 ** 2)
        agent.channels = CHANNELS
        c = candidates[1]
        c.update(prior_precision=0., precision=1e12, weighted=.2e12)
        mean, se = agent._channel_estimate(c, 'call')
        self.assertAlmostEqual(mean - se, .2, places=5)

    def test_malformed_response_accounts_for_spent_resources(self):
        env = PublicEnv()
        original = env.run_pilot
        def malformed(**kwargs):
            original(**kwargs)
            return {'n_customers': 50, 'observed_lift_ratio': float('nan')}
        env.run_pilot = malformed
        with warnings.catch_warnings(record=True) as logged:
            agent, campaigns = self.run_agent(env)
        self.assertTrue(logged)
        self.assertTrue(campaigns)
        self.assertEqual(sum(agent.spent.values()), 15000 - env.remaining_contacts)

    def test_real_package_limits_and_reproducibility(self):
        from local_eval import evaluate_agent
        from make_submission import build_submission
        class CheckedAgent(Agent):
            def act(inner, env):
                campaigns = super(CheckedAgent, inner).act(env)
                self.assertTrue(1 <= len(campaigns) <= 10)
                validate_strategy(pd.DataFrame(campaigns), env.tariffs)
                self.assertTrue(0 < len(env.pilot_history) <= 20)
                self.assertTrue(all(10 <= p['n_customers'] <= 200 for p in env.pilot_history))
                return campaigns
        for seed in range(10):
            result = evaluate_agent(CheckedAgent(), seed=seed, verbose=False)
            self.assertLessEqual(result['total_cost'], 100000)
            self.assertLessEqual(result['total_contacts'], 15000)
            self.assertTrue(all(c['n_contacts'] <= 5000 for c in result['campaigns_detail']))
        expected = build_submission(Agent()).fillna('')
        actual = pd.read_csv('submission.csv').fillna('')
        pd.testing.assert_frame_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()

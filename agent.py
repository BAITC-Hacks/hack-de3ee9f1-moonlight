"""Adaptive, prior-informed tariff campaigns. Uses only the public env API."""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

PILOT_STD = 0.804  # Public measurement noise in environment.py, not a mock effect.
MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000


class Agent:
    def act(self, env) -> list[dict]:
        self.deadline = time.monotonic() + 540
        self.channels = env.channels
        self.spent = {}
        self.pilot_log = []
        self.initial_budget = env.remaining_budget
        self.initial_contacts = env.remaining_contacts
        self.cells, self.overlap = {}, {}
        valid = set(env.tariffs['tariff_plan_code'])
        for key, group in env.customer_profile.groupby(['current_tariff', 'arpu_segment'], observed=True, sort=True):
            if key[0] in valid and key[1] in {'LOW', 'MID', 'HIGH'}:
                arpu = group.sort_values('ID_NUMBER')['predicted_arpu'].to_numpy(float)
                self.cells[key] = np.r_[0., np.cumsum(arpu)]
                self.overlap[key] = np.r_[0., np.cumsum(np.sort(arpu)[::-1])]
        candidates = self._rank_candidates(self._build_prior(), valid)
        # History proposes experiments; uncertainty alone must not promote
        # unobserved/negative transitions above economically plausible ones.
        promising = [c for c in candidates if c['hist_n'] and c['hist_mean'] > 0]
        pending, tested = list(promising or candidates), []
        counts = {'cell': {}, 'target': {}, 'segment': {}}
        while pending and len(tested) < 12 and self._can_explore(env):
            def priority(c):
                diversity = (1 + 2 * counts['cell'].get(c['cell'], 0)
                             + counts['target'].get(c['target'], 0)
                             + counts['segment'].get(c['cell'][1], 0))
                options = [self._option(c, ch, env.remaining_budget, env.remaining_contacts)
                           for ch in sorted(self.channels)]
                value = max((o['expected'] for o in options if o), default=-np.inf)
                return value / diversity
            c = max(pending, key=priority)
            pending.remove(c)
            if self._pilot(c, env, 50, 'explore'):
                tested.append(c)
                for name, key in [('cell', c['cell']), ('target', c['target']), ('segment', c['cell'][1])]:
                    counts[name][key] = counts[name].get(key, 0) + 1

        confirmed = set()
        for _ in range(6):
            if not self._can_explore(env):
                break
            plan = self._select_campaigns(tested, env)
            selected = {o['candidate']['key'] for o in plan}
            cutoff = min((o['value'] for o in plan), default=0) if len(plan) == 10 else 0
            choices = []
            for c in tested:
                if c['key'] in confirmed:
                    continue
                low, high = self._best(c, env), self._best(c, env, optimistic=True)
                if not high or high['value'] <= 0:
                    continue
                rivals = [self._best(r, env, optimistic=True) for r in tested
                          if r['cell'] == c['cell'] and r is not c]
                boundary = max(cutoff, max((r['value'] for r in rivals if r), default=0))
                lower = low['value'] if low else 0
                if c['key'] in selected and not lower <= boundary <= high['value']:
                    continue
                if c['key'] not in selected and high['value'] < boundary:
                    continue
                _, se = self._belief(c)
                size = 200 if se > 0.15 else 100
                # Approximate decision value of reducing the uncertainty interval.
                value = (high['value'] - lower) * (1 - np.sqrt(50 / (50 + size)))
                choices.append((value, c, size))
            if not choices:
                break
            value, c, size = max(choices, key=lambda row: row[0])
            confirmed.add(c['key'])
            opportunity = max((o['value'] / o['reach'] for o in plan), default=0)
            if value <= size * opportunity:
                break
            self._pilot(c, env, size, 'confirm', value)
        plan = self._select_campaigns(tested, env)
        if not plan:
            plan = self._fallback_campaigns(candidates, env)
        self.diagnostics = {'pilots': self.pilot_log, 'final_campaigns': len(plan)}
        return [self._campaign(o['candidate'], o['channel']) for o in plan]

    def _build_prior(self):
        columns = ['tariff_plan_code_from', 'tariff_plan_code_to', 'arpu_segment', 'ratio', 'n', 'std']
        try:
            hist = pd.read_csv(Path(__file__).resolve().parent / 'data/change_tariff.csv')
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            warnings.warn(f'Historical data unavailable: {exc}')
            return pd.DataFrame(columns=columns)
        required = {'tariff_plan_code_from', 'tariff_plan_code_to', 'AVG_ARPU_PREV_3M', 'AVG_ARPU_NEXT_3M'}
        if not required.issubset(hist.columns):
            warnings.warn('Historical schema incomplete; using neutral hypotheses')
            return pd.DataFrame(columns=columns)
        before = pd.to_numeric(hist['AVG_ARPU_PREV_3M'], errors='coerce')
        after = pd.to_numeric(hist['AVG_ARPU_NEXT_3M'], errors='coerce')
        good = np.isfinite(before) & np.isfinite(after) & (before >= 100) & (after >= 0)
        hist, before, after = hist.loc[good].copy(), before[good], after[good]
        hist['arpu_segment'] = np.where(before < 1000, 'LOW', np.where(before <= 5000, 'MID', 'HIGH'))
        hist['ratio'] = ((after - before) / before).clip(-1, 3)
        return (hist.groupby(columns[:3], observed=True, sort=True)
                .agg(ratio=('ratio', 'mean'), n=('ratio', 'size'), std=('ratio', 'std')).reset_index())

    def _rank_candidates(self, prior, valid):
        stats = {(r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to): r for r in prior.itertuples()}
        out = []
        for cell in self.cells:
            for target in sorted(valid):
                if target == cell[0]:
                    continue
                key = (*cell, target)
                r = stats.get(key)
                mean, n, std = (float(r.ratio), int(r.n), float(r.std)) if r else (0., 0, 1.)
                std = std if np.isfinite(std) else 1.
                # Switcher history cannot identify campaign conversion probability.
                # Weak uniform [0,1] conversion assumption: E[p]=.5, Var[p]=1/12.
                weight = min(20., 2 * np.log1p(n))
                variance = std ** 2 / max(n, 1) + 0.20 ** 2 + mean ** 2 / 12
                precision = min(weight / PILOT_STD ** 2, 1 / variance) if n else 0.
                out.append(dict(key=key, cell=cell, target=target, hist_mean=mean, hist_n=n,
                                prior_mean=0.5 * mean, prior_precision=precision,
                                precision=0., weighted=0., pilots=0))
        return out

    def _belief(self, c):
        precision = c['prior_precision'] + c['precision']
        if precision == 0:
            return 0., 0.5
        mean = (c['prior_mean'] * c['prior_precision'] + c['weighted']) / precision
        return mean, np.sqrt(1 / precision)

    def _channel_estimate(self, c, channel):
        mean, se = self._belief(c)
        multiplier = self.channels[channel]['conversion_multiplier']
        if multiplier <= 1:
            return mean * multiplier, se * multiplier
        # delta*min(p*m,1): relative to delta*p, scale lies in [1,m].
        # Treat that unidentified saturation interval as additional uncertainty.
        return mean * (1 + multiplier) / 2, se * multiplier + abs(mean) * (multiplier - 1) / 2

    def _option(self, c, channel, budget, contacts, optimistic=False):
        cost = self.channels[channel]['cost_per_contact']
        sums = self.cells[c['cell']]
        reach = min(len(sums) - 1, MAX_CUSTOMERS_PER_CAMPAIGN, int(contacts))
        if cost > 0:
            reach = min(reach, int(budget // cost))
        if reach <= 0:
            return None
        mean, se = self._channel_estimate(c, channel)
        overlap = min(self.spent.get(c['cell'], 0), reach)
        # Worst-case ARPU overlap: pilot IDs are not part of the public API.
        revenue = max(0., sums[reach] - self.overlap[c['cell']][overlap])
        ratio = mean + se if optimistic else mean - se
        # For expected value, random pilot sampling permits an overlap estimate.
        # Keep the worst-case discount for the risk score, and never discount
        # negative effects (that would make a harmful campaign look safer).
        expected_revenue = sums[reach] * max(0., 1 - self.spent.get(c['cell'], 0) / (len(sums) - 1))
        if ratio < 0:
            revenue = sums[reach]
        if mean < 0:
            expected_revenue = sums[reach]
        return dict(candidate=c, channel=channel, reach=reach, cost=reach * cost,
                    value=revenue * ratio - reach * cost, expected=expected_revenue * mean - reach * cost)

    def _best(self, c, env, optimistic=False):
        options = [self._option(c, ch, env.remaining_budget, env.remaining_contacts, optimistic)
                   for ch in sorted(self.channels)]
        return max((o for o in options if o), key=lambda o: o['value'], default=None)

    def _can_explore(self, env):
        return (time.monotonic() < self.deadline and env.pilots_left > 0
                and self.initial_contacts - env.remaining_contacts < min(2000, self.initial_contacts - 1))

    def _pilot(self, c, env, requested, stage, information_value=float('inf')):
        size = min(requested, len(self.cells[c['cell']]) - 1, env.remaining_contacts - 1,
                   2000 - (self.initial_contacts - env.remaining_contacts))
        if size < 10 or not self._can_explore(env):
            return False
        money = min(env.remaining_budget, 0.15 * self.initial_budget - (self.initial_budget - env.remaining_budget))
        choices = []
        for channel, spec in sorted(self.channels.items()):
            # Pilots use unsaturated channels to identify the shared delta*p.
            if spec['conversion_multiplier'] > 1:
                continue
            pilot_cost = size * spec['cost_per_contact']
            if pilot_cost <= money and pilot_cost <= information_value:
                option = self._option(c, channel, env.remaining_budget, env.remaining_contacts)
                if option:
                    choices.append(option)
        if not choices:
            return False
        channel = max(choices, key=lambda o: o['expected'])['channel']
        pilot_cost = size * self.channels[channel]['cost_per_contact']
        if env.remaining_budget < pilot_cost or env.remaining_contacts < size or env.pilots_left <= 0:
            return False
        before = env.remaining_contacts
        try:
            result = env.run_pilot(target_tariff=c['target'], channel=channel, n_customers=size,
                                   filter_current_tariff=c['cell'][0], filter_arpu_segment=c['cell'][1])
        except (RuntimeError, ValueError, OSError) as exc:
            warnings.warn(f"Pilot failed for {c['key']}: {exc}")
            return False
        finally:
            self.spent[c['cell']] = self.spent.get(c['cell'], 0) + max(0, before - env.remaining_contacts)
        try:
            n, observed = int(result['n_customers']), float(result['observed_lift_ratio'])
            if not 10 <= n <= size or not np.isfinite(observed):
                raise ValueError('Invalid pilot observation')
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            warnings.warn(f'Malformed pilot response: {exc}')
            return False
        multiplier = self.channels[channel]['conversion_multiplier']
        precision = n * multiplier ** 2 / PILOT_STD ** 2
        c['precision'] += precision
        c['weighted'] += precision * observed / multiplier
        c['pilots'] += 1
        self.pilot_log.append(dict(stage=stage, key=c['key'], channel=channel, n=n, observed=observed))
        return True

    def _select_campaigns(self, candidates, env):
        # Search budget shadow prices to balance expensive reach against cheap reach.
        base = [self._option(c, ch, env.remaining_budget, env.remaining_contacts)
                for c in candidates for ch in sorted(self.channels)]
        ratios = [o['value'] / o['cost'] for o in base if o and o['value'] > 0 and o['cost'] > 0]
        penalties = [0.] + (list(np.quantile(ratios, np.linspace(0, 1, 9))) if ratios else [])
        best_plan, best_value = [], 0.
        for penalty in penalties:
            budget, contacts = env.remaining_budget, env.remaining_contacts
            used, plan = set(), []
            while len(plan) < MAX_CAMPAIGNS and contacts > 0:
                options = [self._option(c, ch, budget, contacts) for c in candidates
                           if c['cell'] not in used for ch in sorted(self.channels)]
                options = [o for o in options if o and o['value'] > 0 and o['expected'] > 0]
                if not options:
                    break
                chosen = max(options, key=lambda o: o['value'] - penalty * o['cost'])
                plan.append(chosen)
                used.add(chosen['candidate']['cell'])
                budget -= chosen['cost']
                contacts -= chosen['reach']
            value = sum(o['value'] for o in plan)
            if value > best_value:
                best_plan, best_value = plan, value
        return best_plan

    def _fallback_campaigns(self, candidates, env):
        if not self.channels:
            return []
        cheapest = min(self.channels, key=lambda ch: (self.channels[ch]['cost_per_contact'], ch))
        options = [self._option(c, cheapest, env.remaining_budget, env.remaining_contacts)
                   for c in candidates if c['hist_n'] > 0 and c['hist_mean'] > 0]
        options = [o for o in options if o and o['expected'] > 0]
        return [max(options, key=lambda o: o['value'])] if options else []

    @staticmethod
    def _campaign(c, channel):
        return {'campaign_name': f"{c['cell'][0]}_to_{c['target']}_{c['cell'][1]}_{channel}",
                'filter_current_tariff': c['cell'][0], 'filter_arpu_segment': c['cell'][1],
                'target_tariff': c['target'], 'channel': channel}

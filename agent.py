"""
Beeline Tariff Marketing Campaigns — Agent

Подход:
1. ПРИОР: считаем средний относительный эффект (current_tariff -> target_tariff,
   arpu_segment) из выданной истории data/change_tariff.csv — "как обычно бывает".
2. РАНЖИРОВАНИЕ: комбинируем приор с размером и ценностью сегмента в текущей
   аудитории (customer_profile), чтобы не тратить пилоты на крошечные/дешёвые
   сегменты. Каждому сегменту сразу назначается канал по эвристике ценности
   (HIGH ARPU -> call, MID -> digital_ads, LOW -> sms) — пилотируем и потом
   выкатываем кампанию ОДНИМ и тем же каналом, чтобы не пришлось пересчитывать
   эффект под другой канал.
3. ПИЛОТЫ: проверяем топ-кандидатов пилотами (env.run_pilot).
4. POSTERIOR: комбинируем приор и наблюдение байесовским сглаживанием — чем
   больше людей в пилоте, тем больше доверия наблюдению, а не истории.
5. ЖАДНЫЙ ВЫБОР: сортируем по чистому эффекту на абонента, набираем до 10
   кампаний без пересечения по сегментам, укладываясь в остаток бюджета/охвата.
6. Fallback: если пилоты не дали ни одной прибыльной связки (неудачная серия),
   берём лучшие кандидаты по приору с бесплатным каналом push — нулевая
   стоимость контакта гарантирует неотрицательный чистый результат по знаку
   приора, даже без пилотных подтверждений.
7. Всё обёрнуто в try/except: агент не должен падать в ноль — уже проведённые
   пилоты идут в зачёт независимо от исхода.
"""

import numpy as np
import pandas as pd

SEGMENT_CHANNEL = {"HIGH": "call", "MID": "digital_ads", "LOW": "sms"}
FALLBACK_CHANNEL = "sms"

PILOT_SIZE = 150          # компромисс точность/цена (std ~0.065 при n=150)
MAX_CANDIDATES = 14       # сколько связок пилотируем максимум (лимит пилотов = 20)
MIN_SEGMENT_SIZE = 80     # не тратим пилот на слишком маленький сегмент
PRIOR_PSEUDOCOUNT = 60    # "вес" приора в эквиваленте клиентов при сглаживании
MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000


class Agent:
    def act(self, env) -> list[dict]:
        self._partial_campaigns = []
        try:
            return self._run(env)
        except Exception as e:
            print(f"[agent] упал с ошибкой, возвращаю то, что успели: {e}")
            return self._partial_campaigns

    # ------------------------------------------------------------------ #

    def _run(self, env):
        profile = env.customer_profile
        channels = env.channels

        prior = self._build_prior()
        candidates = self._rank_candidates(profile, prior)

        observations = []
        for cur_tariff, seg, target, channel, prior_ratio, prior_n in candidates[:MAX_CANDIDATES]:
            if env.pilots_left <= 0:
                break
            cost_per_contact = channels[channel]["cost_per_contact"]
            if cost_per_contact > 0 and env.remaining_budget < cost_per_contact * 30:
                continue
            try:
                res = env.run_pilot(
                    target_tariff=target, channel=channel, n_customers=PILOT_SIZE,
                    filter_arpu_segment=seg, filter_current_tariff=cur_tariff,
                )
            except (RuntimeError, ValueError):
                continue
            observations.append({
                "current_tariff": cur_tariff, "arpu_segment": seg,
                "target_tariff": target, "channel": channel,
                "observed_ratio": res["observed_lift_ratio"],
                "n": res["n_customers"], "prior_ratio": prior_ratio,
            })

        campaigns = self._select_campaigns(observations, profile, channels, env)

        if not campaigns:
            campaigns = self._fallback_campaigns(candidates, env)

        self._partial_campaigns = campaigns
        return campaigns

    # ------------------------------------------------------------------ #

    def _build_prior(self) -> pd.DataFrame:
        hist = pd.read_csv("data/change_tariff.csv")
        hist = hist[hist["AVG_ARPU_PREV_3M"] >= 100].copy()
        hist["arpu_segment"] = pd.cut(
            hist["AVG_ARPU_PREV_3M"], bins=[-np.inf, 1000, 5000, np.inf],
            labels=["LOW", "MID", "HIGH"])
        hist["ratio"] = ((hist["AVG_ARPU_NEXT_3M"] - hist["AVG_ARPU_PREV_3M"])
                          / hist["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        grouped = (hist.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"],
                                 observed=True)
                   .agg(ratio=("ratio", "mean"), n=("ID_NUMBER", "size"))
                   .reset_index())
        return grouped

    def _rank_candidates(self, profile, prior):
        pop = (profile.groupby(["current_tariff", "arpu_segment"], observed=True)
               .agg(pop=("ID_NUMBER", "size"), avg_arpu=("predicted_arpu", "mean"))
               .reset_index())

        cand = prior.merge(
            pop, left_on=["tariff_plan_code_from", "arpu_segment"],
            right_on=["current_tariff", "arpu_segment"], how="inner")
        cand = cand[cand["tariff_plan_code_from"] != cand["tariff_plan_code_to"]]
        cand = cand[cand["pop"] >= MIN_SEGMENT_SIZE]
        cand = cand[cand["ratio"] > 0]
        if cand.empty:
            return []

        cand["channel"] = cand["arpu_segment"].map(SEGMENT_CHANNEL).fillna(FALLBACK_CHANNEL)
        cand["score"] = cand["ratio"] * cand["avg_arpu"] * np.minimum(cand["pop"], MAX_CUSTOMERS_PER_CAMPAIGN)
        cand = cand.sort_values("score", ascending=False)

        out = []
        for _, r in cand.iterrows():
            out.append((r["tariff_plan_code_from"], r["arpu_segment"], r["tariff_plan_code_to"],
                        r["channel"], float(r["ratio"]), int(r["n"])))
        return out

    def _select_campaigns(self, observations, profile, channels, env):
        if not observations:
            return []
        pop = (profile.groupby(["current_tariff", "arpu_segment"], observed=True)
               .agg(pop=("ID_NUMBER", "size"), avg_arpu=("predicted_arpu", "mean"))
               .reset_index())

        scored = []
        for obs in observations:
            row = pop[(pop["current_tariff"] == obs["current_tariff"]) &
                      (pop["arpu_segment"] == obs["arpu_segment"])]
            if row.empty:
                continue
            avg_arpu = float(row["avg_arpu"].iloc[0])
            seg_pop = int(row["pop"].iloc[0])
            n = obs["n"]
            posterior_ratio = (obs["prior_ratio"] * PRIOR_PSEUDOCOUNT + obs["observed_ratio"] * n) \
                / (PRIOR_PSEUDOCOUNT + n)
            cost_per_contact = channels[obs["channel"]]["cost_per_contact"]
            net_per_customer = posterior_ratio * avg_arpu - cost_per_contact
            if net_per_customer <= 0:
                continue
            scored.append({
                **obs, "net_per_customer": net_per_customer,
                "reach": min(seg_pop, MAX_CUSTOMERS_PER_CAMPAIGN),
            })

        scored.sort(key=lambda x: x["net_per_customer"], reverse=True)

        campaigns = []
        used_segments = set()
        remaining_budget = env.remaining_budget
        remaining_contacts = env.remaining_contacts
        for s in scored:
            if len(campaigns) >= MAX_CAMPAIGNS:
                break
            key = (s["current_tariff"], s["arpu_segment"])
            if key in used_segments:
                continue
            cost_per_contact = channels[s["channel"]]["cost_per_contact"]
            max_by_money = remaining_contacts if cost_per_contact == 0 else \
                min(remaining_contacts, int(remaining_budget // cost_per_contact))
            reach = min(s["reach"], max_by_money)
            if reach <= 0:
                continue
            campaigns.append({
                "campaign_name": f"{s['current_tariff']}_to_{s['target_tariff']}_{s['channel']}",
                "filter_arpu_segment": s["arpu_segment"],
                "filter_current_tariff": s["current_tariff"],
                "target_tariff": s["target_tariff"],
                "channel": s["channel"],
            })
            used_segments.add(key)
            remaining_contacts -= reach
            remaining_budget -= reach * cost_per_contact

        return campaigns

    def _fallback_campaigns(self, candidates, env):
        """Пилоты не дали прибыльных связок (неудачная серия) — берём лучшие по
        приору с бесплатным каналом push, чтобы не сдать пустой список."""
        campaigns = []
        used_segments = set()
        for cur_tariff, seg, target, _channel, prior_ratio, _n in candidates:
            if prior_ratio <= 0 or len(campaigns) >= 5:
                continue
            key = (cur_tariff, seg)
            if key in used_segments:
                continue
            campaigns.append({
                "campaign_name": f"fallback_{cur_tariff}_to_{target}_push",
                "filter_arpu_segment": seg,
                "filter_current_tariff": cur_tariff,
                "target_tariff": target,
                "channel": "push",
            })
            used_segments.add(key)
        return campaigns

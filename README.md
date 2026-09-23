# Beeline Tariff Marketing Campaigns Agent

The agent helps a marketing analyst choose tariff offers and communication
channels under uncertain customer response. It maximizes an estimate of net
incremental ARPU within the competition's contact, campaign, and money limits.
All supplied data are synthetic. Historical switchers are a different population
from the campaign audience, so historical gains are hypotheses, not promises.

## Architecture and why this is agentic

The existing Python/pandas/numpy solution is extended through `Agent.act(env)`.
There are no external services, LLM calls, heavy models, or additional runtime
dependencies. It reads the public customer profile and tariff/channel dictionaries,
uses the supplied historical CSV, and interacts only with the documented pilot API.

```text
Historical data
      ↓
Prior hypotheses
      ↓
Candidate generation
      ↓
Small exploratory pilots
      ↓
Uncertainty + expected value analysis
      ↓
Selective confirmation pilots
      ↓
Belief update
      ↓
Budget-aware campaign selection
      ↓
Final campaign plan
```

The agent does not simply predict from historical data. It performs experiments,
observes results, updates its beliefs, and changes subsequent experiments and
campaign choices based on those observations. Candidate ordering and tie breaking
are deterministic; the environment supplies seeded pilot randomness.

## Historical prior and candidate generation

`_build_prior()` groups `data/change_tariff.csv` by current tariff, target tariff,
and ARPU segment. It reports relative-lift mean, sample count, and sample standard
deviation. Nonfinite observations, negative after-ARPU, and before-ARPU below 100
are excluded. Relative changes are clipped to [-1, 3] so tiny denominators and
extreme observations cannot dominate. Segments are LOW <1000, MID 1000–5000,
HIGH >5000. Current audience labels come directly from `env.customer_profile`.

The historical sample contains switchers, not randomized campaign recipients.
It cannot identify campaign conversion probability. The agent deliberately uses
a weak uniform [0,1] conversion assumption with mean 0.5 and variance 1/12.
This assumption is overridden by pilots; historical transition frequency is not
treated as the hidden environment's conversion probability.

For historical mean `h`, standard deviation `s`, and count `n`:

```text
w = min(20, 2 * log(1 + n))
prior_mean = 0.5 * h
prior_variance = s² / max(n, 1) + 0.20² + h² / 12
prior_precision = min(w / 0.804², 1 / prior_variance)
```

The 20-observation cap and 0.20 population-shift allowance deliberately keep
history weak. Five and 500 historical observations receive different confidence,
but even abundant history cannot overwhelm pilots. Singleton standard deviations
default to 1. Missing history has zero precision and a neutral exploration estimate.
These are conservative modeling choices, not constants fitted to mock seeds.

Candidates pair each observed current-tariff/ARPU cell with each valid different
target tariff. Stage 1 prioritizes historically positive hypotheses by expected
total economic value. Missing-history hypotheses are available when no positive
historical candidates exist. A diversity penalty discourages repeating the same
cell, target, or ARPU segment; it does not force uneconomic segment coverage.

## Adaptive exploration and uncertainty

Stage 1 runs up to 12 successful exploratory pilots of 50 people (smaller only
when audience or resources require it, never below 10). Skipped and failed
candidates do not count toward that success target. Channel choice compares
expected net value at reachable audience size, subject to exploration resources.

Stage 2 considers up to six follow-up hypotheses. It recomputes the campaign plan,
checks whether optimistic/pessimistic values straddle zero, a competing offer in
the same cell, or the ten-campaign selection boundary, and estimates the value of
reducing that uncertainty. Follow-ups use 100 people, or 200 when base-effect
standard error exceeds 0.15. It stops when no decision is ambiguous, information
value cannot justify contact opportunity cost, or resources/time are exhausted.
It does not automatically consume all 20 pilots.

The actual `run_pilot()` result contains `observed_lift_ratio`,
`observed_lift_total`, `n_customers`, `cost`, campaign/channel identifiers and
remaining resources. It contains neither customer IDs nor a confidence interval.
`observed_lift_total` derives from the same noisy ratio and is not independent
evidence. The public environment documents per-person measurement noise 0.804,
giving ratio standard error `0.804 / sqrt(n)`.

Pilots use channels with multiplier `m <= 1`. They measure the common base effect
`x = tariff_change * conversion_probability` without conversion saturation:

```text
pilot_base_estimate = observed_lift_ratio / m
pilot_precision = n * m² / 0.804²
updated_mean = (prior_precision * prior_mean
                + sum(pilot_precision * pilot_base_estimate))
               / (prior_precision + sum(pilot_precision))
updated_SE = sqrt(1 / (prior_precision + sum(pilot_precision)))
```

This is **prior-informed shrinkage**, not full Bayesian inference. Larger samples
receive more weight. Stage 1 and stage 2 observations accumulate rather than
replacing one another. Decisions penalize one standard error; this is an
explainable risk score, not a calibrated confidence guarantee after selection.

## Economic channel selection

The organizer's actual mechanics are:

```text
lift_ratio = tariff_change * min(conversion_probability * channel_multiplier, 1)
net = sum(predicted_arpu * lift_ratio for credited customers) - contact_cost
```

Costs and multipliers come from `env.channels`. All valid channels are compared
for final campaigns. Push/SMS/digital ads scale the base estimate exactly under
the documented probability semantics. For call (multiplier 1.2), the scale lies
between 1 and 1.2 depending on unknown saturation. The estimate uses the interval
midpoint and adds its half-width to the risk penalty. It never assumes call gives
an unconditional 1.2x benefit. Call must justify both its cost and reduced reach;
HIGH ARPU does not automatically imply call.

## Final campaign optimization and overlap

`_select_campaigns()` ranks **total** risk-adjusted net value over the reachable
audience, not merely net value per customer. It evaluates greedy plans at several
budget shadow prices derived from current value/cost ratios and retains the plan
with highest total risk-adjusted value. This makes cheap broad reach compete with
expensive concentrated reach. It is a heuristic, not an exact global optimizer.

The planner reproduces public scoring order: exact tariff/ARPU filters, ascending
`ID_NUMBER`, first 5,000 people, then remaining contact and money caps. Prefix
sums use the actual selected audience ARPU. No unsupported final audience-size
field is emitted; the scorer applies these caps in campaign order.

At most one final campaign is selected for each current-tariff/ARPU cell, making
final filters disjoint. Pilot IDs are unavailable, so pilot/final overlap cannot
be removed exactly. The risk score subtracts an upper bound on possible overlap
ARPU (the largest ARPUs for the number of pilot contacts in the cell). Expected
value uses a conservative random-sampling overlap estimate. Negative effects are
never discounted away. All repeated contacts still consume budget and reach;
the organizer credits each customer only once, by their best campaign.

## Fallback, limits, and failures

If no risk-positive plan remains, the fallback returns one historically promising
offer on the cheapest valid channel, provided its updated expected net value is
positive. This **minimizes downside under uncertainty**; push does not guarantee
profit. It does not override clearly negative pilot evidence or pad the plan to
ten campaigns. If resources are exhausted or every feasible estimate is
nonpositive, it returns no campaign rather than knowingly returning a loss.

Limits: 10 final campaigns, 5,000 people per campaign, 15,000 total contacts,
100,000 total communication budget, 20 pilots, 10–200 people per pilot. Exploration
uses at most 15% of starting money and 2,000 contacts, while reserving at least
one contact for the final plan. Before **every** pilot, the agent checks actual
`pilot_size * channel_cost`, remaining contacts and pilots. Planning starts from
the environment's remaining resources, so pilots are included in every limit.

Environment calls catch `RuntimeError`, `ValueError`, and `OSError` locally and
warn before continuing to another candidate. Malformed/nonfinite responses are
rejected. Resources consumed by a failed/malformed call are still accounted for.
Historical-file read/schema failures produce neutral pilot hypotheses. Programming
errors are not hidden by a blanket exception handler. The misleading partial-plan
state has been removed. No private environment state or hidden model is accessed.

## Running and evaluating

Use Python 3.12 with the supplied participant package in this directory:

```sh
python -m pip install -r requirements.txt
python local_eval.py
python local_eval.py --runs 10
python make_submission.py
python -m unittest -v test_agent
```

On this Windows workspace, Python is installed locally; use
`.venv\Scripts\python.exe` in place of `python`. Tested with numpy 2.5.3 and
pandas 3.0.6. The agent needs only numpy and pandas. The unit/integration suite
also verifies ten seeds' limits, API failures, malformed responses, small budgets,
negative-effect belief reversal, prior confidence, call saturation, and exact
submission reproduction. Tests require the generated `submission.csv`.

Submit `agent.py`, `submission.csv`, and `requirements.txt`; include this README
for explanation. Organizer environment, scoring, evaluation, templates, and
submission-generation files are preserved byte-for-byte from the supplied ZIP.

## Actual local results

Final single run (`seed=42`): net **910,618**, gross **972,300**, communication
cost **61,682 / 100,000**, contacts **4,547 / 15,000**, **17 pilots**, **6 final
campaigns**, status PASS. The evaluator prints 23 campaigns because its total
includes 17 pilots plus 6 final campaigns. Its remaining-budget/contacts lines
describe the state after pilots; the total-cost/contact lines include finals.

Final `python local_eval.py --runs 10` output (rounded by the organizer command):

| Seed | Net gain |
| --- | ---: |
| 0 | 744,304 |
| 1 | 1,370,832 |
| 2 | 1,490,558 |
| 3 | 1,351,624 |
| 4 | 847,616 |
| 5 | 1,464,192 |
| 6 | 956,181 |
| 7 | 714,869 |
| 8 | 939,769 |
| 9 | 1,509,310 |

**10/10 positive**; median **1,153,902**; minimum **714,869**; maximum **1,509,310**.
`make_submission.py` successfully generated six campaigns.
All eight unit/integration tests passed in 17.2 seconds, including ten complete
seed evaluations and submission reproduction; runtime was well below 10 minutes.

**The mock environment validates mechanics and robustness; judging effects are different.**

## Known limitations

- Historical positive-offer screening can miss transitions whose sign reverses
  favorably in the judging population. Diversity is a heuristic, not exhaustive
  exploration of every tariff/segment.
- Weak conversion and population-shift priors remain assumptions; the public
  noise model and normal approximation may not describe a changed environment.
- Call saturation is bounded, not independently identified with call pilots.
- Pilot overlap is conservative because identities are not exposed. Follow-up
  pilots can resample customers; repeated contacts are not new unique reach.
- One-standard-error ranking can still select false positives. No profitability
  guarantee follows from ten positive mock seeds or from the fallback.
- Global planning and information value are approximations. The agent does not
  optimize separate data/call subsegments or split one cell over channels.
- Exploration checks a 540-second deadline; a blocking external implementation
  of `run_pilot` cannot be interrupted through the documented interface.

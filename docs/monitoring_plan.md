# Monitoring plan: main PD model (illustrative)

This is the written monitoring plan for the Stage 5 logistic regression PD model and the Stage 7
illustrative risk grades. It belongs to an educational project. It is **not** a bank's model-risk
policy and claims no regulatory compliance. Every threshold is a HEURISTIC chosen for this project
(decision log [D-028](decision_log.md)), not an industry or regulatory standard.

Code: `src/monitoring.py` (`python -m src.monitoring`), `sql/monitoring_bin_counts.sql`,
`sql/monitoring_psi.sql`. Results: `artifacts/monitoring_*.csv`, `notebooks/05_monitoring.ipynb`.

## 1. What this plan can and cannot show with this dataset

- `year` is 2019 for every loan (D-004). There is no later period, so **there is no drift to
  detect**.
- The hold-out stands in for a "next period". It is a random, stratified 30% of the same
  population (D-013), so its PSI against development is close to zero by construction. The
  expected value with no shift is about (bins − 1) × (1/n₁ + 1/n₂), roughly 0.0003 here.
- A green run therefore shows that the monitoring code works and passes a sanity check. **It is not
  evidence that the model is stable over time.**
- No defaults arrive after development, so the outcome-based checks below (backtesting) are
  specified but cannot be run for real.

## 2. Scope

- **Model:** the frozen Stage 5 pipeline (`artifacts/pd_model.joblib`), with 6 features (D-022).
- **Population:** loans outside `credit_type = EQUI` (D-026). EQUI loans are not scored. Their
  share is monitored as a scope KPI.
- **Baseline:** in-scope development loans (93,391), with bins frozen on them and stored in
  DuckDB as `monitoring_baseline`.
- **Data rule (D-025):** a monitoring run reads the monitored sample's inputs and scores only.
  Outcomes enter only through the backtest, and only once they exist for a new period. The
  project's hold-out outcomes are never used here.

## 3. Frequency (illustrative)

- **Monthly:** population stability (PSI, CSI), the watch list and the scope KPI, on new
  applications.
- **Quarterly, or once defaults over the outcome window are known:** the per-grade backtest.
- **Annually:** a full review that repeats the Stage 6 validation on newer data.

These frequencies show how a monitoring cycle would be set up. Nothing in this dataset arrives
over time.

## 4. Metrics

| Metric | What it answers | Bins / method | Light |
|---|---|---|---|
| Score PSI | Has the PD distribution moved? | 10 equal-count development bins (edges frozen) | ≤ 0.10 green, ≤ 0.25 amber, > 0.25 red |
| Grade PSI and grade shares | Has the mix of grades moved? | the 8 D-027 grades | same |
| CSI per model feature | Which input has moved? | numeric features: 10 development deciles + `<missing>`; categorical features: development levels + `<missing>` + `<unseen>` | same |
| χ² reference | Is a PSI larger than sampling noise? | PSI × n₁n₂/(n₁+n₂) ≈ χ²(bins − 1) | reported only |
| Characteristic analysis | Which feature moves the mean score? | coefficient × change in mean transformed value | reported only |
| Per-grade backtest | Does each grade default at its grade PD? | binomial test and gap per grade | gap ≤ 2 pp green, ≤ 5 pp amber; binomial p ≥ 0.05 green, ≥ 0.01 amber (D-025 thresholds) |

Shares are floored at 0.0001 before the logarithm, so an empty bin gives a large but finite PSI.

## 5. Watch list: what to watch first

These are the model's known weak points from validation and grading.

| Priority | KPI | Why | Baseline (development) | Trigger |
|---|---|---|---|---|
| 1 | Share of loans in the top development `loan_amount` decile bin, and in the top `income_clean` decile bin | D-023: the model under-predicts the top deciles by 3.4 pp and 3.5 pp. More such loans means more under-prediction in the portfolio. | 10.1% and 9.4% | > 15% amber, > 20% red |
| 1 | *Once outcomes exist:* observed minus predicted default rate in those top deciles | D-023 | hold-out: +3.4 pp and +3.5 pp (Stage 6) | > 5 pp red (the D-025 amber limit) |
| 2 | Change in grade A's share | D-027: 45% of loans sit in one grade, where the model cannot rank risk. A larger grade A means less differentiation. | 45.0% | change > 5 pp amber, > 10 pp red |
| 2 | *Once outcomes exist:* grade A's observed rate against its grade PD (9.5%) | D-027 | development 9.6% | gap > 2 pp amber, > 5 pp red |
| 3 | Out-of-scope (EQUI) share of all loans | D-026: these loans get no PD. A rising share means a growing unscored part of the portfolio. | 10.26% | > 15% amber, > 20% red |
| 4 | Share of `lump_sum_payment = lpsm` | D-023: 1.65% of loans, close to the 1% rare-level rule | 1.65% | < 1% amber |
| 4 | Loans with a categorical level unseen in development | D-023: the encoder rejects unseen levels, so such loans cannot be scored | 0 | > 0 red |
| 5 | `income_clean` missing share | D-023: 7.0% missing, imputed with a missing-value indicator | 7.0% | covered by the CSI `<missing>` bin |

## 6. Actions

| Light | Action |
|---|---|
| Green | No action. Record the run. |
| Amber | Investigate: find which feature drives the change (CSI, characteristic analysis) and whether the change is a business change or a data problem. Document the finding. Run again at the next cycle. |
| Red | Escalate to the model owner (Bruno). Consider recalibration (for example of the intercept) or redevelopment. Any change to the model or the grades needs approval and a decision-log entry (D-025). |
| Unseen level | Stop scoring the affected loans until the level is mapped. Mapping a level is a modelling decision. |

## 7. What would change with real time-stamped data

- The baseline would stay frozen at development, and each period's applications would become the
  monitored sample (the `monitored_sample` argument of `run_stage8`).
- Backtesting would compare each grade's realised default rate over the outcome window with its
  grade PD, and the top-decile gaps with the D-023 limit.
- Out-of-time validation (D-004) would become possible and would replace the random hold-out as the
  main performance check.

## 8. First run (development vs hold-out, inputs only)

- Every PSI and CSI is green, between 0.0000 and 0.0006. The score's PSI is 0.0004, against 0.0003
  expected from noise alone. No χ² p-value is below 0.05.
- Watch list: all green.
  - top-decile shares: 9.8% and 9.3% in the hold-out;
  - grade A: 44.9%;
  - out-of-scope share: 10.36%;
  - `lpsm`: 1.70%; no unseen levels.
- This is the result a random split must give. It is a sanity check, not evidence of stability.

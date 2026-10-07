"""Stage 9: the dashboard's explanatory sentences (app.py; decision log D-029, D-030).

Short labels and settings live in src/config.py; the longer sentences live here so
config stays readable. No result number is typed here: every number shown on the
dashboard is read from a committed CSV or from config, and templates such as
FINDINGS are filled with those values by src/dashboard.py.
"""

from src import config

DISCLAIMER = "Learning project. Not a credit decision or an official scoring system."

# ---------------------------------------------------------------------------
# Glossary: tile and column help, one or two sentences per term
# ---------------------------------------------------------------------------

GLOSSARY = {
    "pd": "The model's estimate of the chance that a loan defaults: among many loans with the same "
          "PD, it is the share expected to default. What counts as a default is not documented in "
          "the source data.",
    "samples": "Development loans build the model. Hold-out loans were set aside at the start and "
               "test it on loans it has not seen.",
    "cv": f"The development loans are split into {config.CV_FOLDS} parts. The model is fitted on all "
          "but one part and tested on the remaining part, in turn.",
    "auc": "The chance that the model gives a higher PD to a loan that defaulted than to one that "
           "did not. 0.5 is a coin flip; 1 is perfect ranking.",
    "gini": "AUC rescaled (2 × AUC − 1), so 0 means no ranking power and 1 means perfect ranking.",
    "ks": "The largest gap between how PDs are spread for defaulted and for non-defaulted loans. "
          "Higher means better separation.",
    "brier": "The average squared difference between the PD and the outcome (1 default, 0 not). "
             "Lower is better.",
    "calibration": "Whether PDs match what happened: loans given PDs near a value should default "
                   "at about that rate.",
    "grade": "A PD band labelled from A (lowest risk) upward. Illustrative for this project.",
    "psi": "How much the share of loans across bins differs between two samples. Near zero means "
           "little shift.",
    "light": "A green / amber / red rating against a threshold set in advance. The thresholds are "
             "heuristics, not a regulatory standard.",
    "leakage": "Information that would not be known when the loan is decided, or that gives the "
               "outcome away.",
    "scope": f"All loans except credit type {config.EQUI_LEVEL}, which the model does not cover.",
}

# ---------------------------------------------------------------------------
# Page text: a one-line subtitle, and each chart's title with its "how to read it" help
# ---------------------------------------------------------------------------

PAGES = {
    "overview": {
        "subtitle": "A probability-of-default (PD) model built as a learning project on a public "
                    "loan-level dataset. Every number here was produced by the project's pipeline.",
        "charts": {
            "hero": ("Cross-validated AUC: main model vs leakage models",
                     "Higher is better ranking. The leakage models reach a near-perfect AUC only "
                     "because they learn when fields are missing, not borrower risk. "
                     + GLOSSARY["auc"]),
        },
    },
    "leakage": {
        "subtitle": "Some fields are missing almost only for loans that defaulted, so a model that "
                    "uses them looks nearly perfect.",
        "charts": {
            "indicators": ("Default rate when a field is missing",
                           "For each field, the two dots compare the default rate of loans where it is "
                           "missing with loans where it is present. The dashed line is the default "
                           "rate of all loans. The same fields as the README figure, except "
                           "`income = 0`, which no committed table holds; "
                           f"{config.EQUI_LEVEL} has one dot, because no committed table holds the "
                           "default rate of the other credit types."),
        },
    },
    "model": {
        "subtitle": f"The main model: a logistic regression on {len(config.MAIN_MODEL_FEATURES)} "
                    "fields, fitted on in-scope development loans.",
        "charts": {
            "coefficients": ("What raises and lowers the PD",
                             "Bars to the right raise the PD, bars to the left lower it, with the "
                             "other fields held fixed. Amounts are logged and standardised; each "
                             "category is compared with its most common level. Hover for the "
                             "technical name, odds ratio and p-value."),
        },
    },
    "validation": {
        "subtitle": "How the frozen model performs on hold-out loans that were not used to build it.",
        "charts": {
            "calibration": ("Calibration by PD decile",
                            "Each point is a tenth of the hold-out loans, ordered by PD. Points near "
                            "the diagonal mean the PDs match what happened. " + GLOSSARY["calibration"]),
            "misses": ("Where the model misses",
                       "The model's mean PD against the observed default rate on hold-out loans: by "
                       "decile of loan amount or income (lines), or by level of loan purpose or type "
                       "(dots). The label marks the largest gap (observed minus PD), in percentage "
                       "points. Both views were pre-set checks before the hold-out was scored."),
        },
    },
    "grades": {
        "subtitle": "Loans grouped by PD into grades, from A (lowest risk) upward.",
        "charts": {
            "rates": ("Grade PD vs observed default rate",
                      "For each grade, its average PD and the share of its development loans that "
                      "defaulted. The crosses are the same check on loans the model did not see "
                      "while being fitted (out-of-fold). Move the slider to see which grade a PD "
                      "falls in."),
            "shares": ("Share of loans per grade",
                       "How the development loans spread over the grades. A large first grade means "
                       "the model cannot separate risk among the lowest PDs."),
        },
    },
    "monitoring": {
        "subtitle": "Whether the mix of loans and scores differs between the development sample "
                    "and the hold-out.",
        "charts": {
            "psi": ("PSI: observed vs expected with no shift",
                    "For each variable, the PSI between development and hold-out (blue) next to the "
                    "PSI that sampling noise alone would give with no real shift (grey). The "
                    f"heuristic limits, green up to {config.PSI_THRESHOLDS[0]} and amber up to "
                    f"{config.PSI_THRESHOLDS[1]}, lie far to the right of this scale. " + GLOSSARY["psi"]),
            "bins": ("Share of loans per bin",
                     "For the chosen variable, the bars compare the share of loans in each bin in "
                     "the development sample and in the hold-out."),
        },
    },
}

# The caveats that change how a result should be read (D-030): (box, text).
CAVEATS = {
    "overview": ("info", f"Loans with credit type {config.EQUI_LEVEL} are outside the model's scope: "
                         "in this data almost all of them defaulted. Model, validation and grade "
                         "figures cover in-scope loans only."),
    "leakage": ("warning", "The cause of this pattern is unknown; it most likely reflects how the "
                           "dataset was assembled. These fields are excluded from the main model and "
                           "appear only on this page."),
    "validation": ("info", "The hold-out was evaluated twice, before and after the model's scope was "
                           "narrowed, so these results are not a fully unbiased estimate. The hold-out "
                           "comes from the same year as the development loans, so this is "
                           "out-of-sample, not out-of-time, validation."),
    "grades": ("info", "These grades are illustrative, built for this project; they are not a bank's "
                       "rating scale, and their number depends on technical settings."),
    "monitoring": ("info", "Both samples come from one random split, so a PSI near zero is expected by "
                           "construction. It shows the monitoring works; it is not evidence of "
                           "stability over time."),
}

# Short captions next to tiles, charts or tables.
EQUI_CAPTION = {
    "leakage": f"The leakage models include {config.EQUI_LEVEL} loans; the main model does not.",
    "model": f"Fitted on in-scope loans only: credit type {config.EQUI_LEVEL} is excluded.",
    "validation": f"In-scope hold-out loans only: credit type {config.EQUI_LEVEL} is excluded.",
}
LOAN_AMOUNT_CAPTION = ("`loan_amount` is positive: for the same income, a larger loan goes with a "
                       "higher PD.")
HOSMER_LEMESHOW_CAPTION = ("Hosmer-Lemeshow is reported without a pass/fail judgement: with this many "
                           "loans it rejects even trivial misfit.")
SENSITIVITY_HELP = ("The same model without "
                    + ", ".join(f"`{f}`" for f in config.SENSITIVITY_DROPPED_FEATURES)
                    + ". Reported only; that field's effect cannot be separated from how the data "
                      "was assembled.")
TAB_LABELS = {"calibration": "Calibration", "misses": "Where the model misses",
              "psi": "PSI", "bins": "Bin shares"}
SIDEBAR_TITLE = "Credit risk PD model"
KEY_FINDINGS = "Key findings"
MODEL_CARD_TITLE = "Model card"
MODEL_FIELDS_TITLE = "Fields in the model"
MISS_SELECTOR = "View"
BIN_SELECTOR = "Variable"
GRADE_SLIDER = ("Try a PD", "Pick a PD to see which illustrative grade it falls in. This is a lookup "
                            "in the committed grade scale; nothing is scored.")
SELECTED_GRADE = "Selected grade"
BACK_TO_START = "Back to"
# One caption above each side panel's tiles: the context the short tile labels share.
PANEL_CAPTIONS = {
    "leakage": "Mean cross-validated AUC",
    "model": "Main model, in-scope development loans",
    "validation": "In-scope hold-out loans",
    "grades": "In-scope development loans",
    "monitoring": "Development vs hold-out",
}

# Help (hover) text per headline tile; ids as in config.TILE_LABELS.
TILE_HELP = {
    "cv_auc_main": GLOSSARY["auc"] + " " + GLOSSARY["cv"],
    "cv_auc_leakage_full": "The main model plus the excluded fields, with their missing values "
                           "visible. Built only to demonstrate the problem. " + GLOSSARY["leakage"],
    "cv_auc_leakage_ablation": "Only whether each excluded field is missing, plus credit type. Built "
                               "only to demonstrate the problem. " + GLOSSARY["leakage"],
    "n_features": "The fields the model uses to estimate a PD.",
    "cv_auc": GLOSSARY["auc"] + " " + GLOSSARY["cv"],
    "cv_auc_without": SENSITIVITY_HELP,
    "holdout_auc": GLOSSARY["auc"],
    "holdout_gini": GLOSSARY["gini"],
    "holdout_ks": GLOSSARY["ks"],
    "holdout_brier": GLOSSARY["brier"],
    "holdout_mean_pd": GLOSSARY["pd"] + " " + GLOSSARY["calibration"],
    "holdout_observed": "Share of in-scope hold-out loans that defaulted. " + GLOSSARY["calibration"],
    "criteria_green": "Validation criteria set before the hold-out was scored. " + GLOSSARY["light"],
    "n_grades": GLOSSARY["grade"],
    "first_grade_share": GLOSSARY["grade"],
    "top_grade_rate": "Share of the highest grade's development loans that defaulted. "
                      + GLOSSARY["grade"],
    "largest_psi": GLOSSARY["psi"] + " " + GLOSSARY["light"],
    "watch_green": "Indicators checked at each monitoring run. " + GLOSSARY["light"],
    "equi_share_holdout": GLOSSARY["scope"],
}

# ---------------------------------------------------------------------------
# Overview: pipeline notes, key findings (templates) and the model card
# ---------------------------------------------------------------------------

# One note per pipeline step (config.PIPELINE_STEPS), under the step's number.
PIPELINE_NOTES = {
    "raw": "loans in the data file",
    "split": "development / hold-out",
    "scope": "development loans in scope",
    "model": f"CV AUC, {len(config.MAIN_MODEL_FEATURES)} fields",
    "validation": "hold-out AUC",
    "grades": "illustrative grades",
    "monitoring": "largest PSI",
}
# Key findings: (title, template). The {placeholders} are filled with CSV values.
FINDINGS = {
    "leakage": ("Leakage found and removed",
                "Missing-value flags alone reach a CV AUC of {ablation}; the main model reaches "
                "{main} without them."),
    "holdout": ("On new loans",
                "Hold-out AUC {auc}; mean PD {mean_pd} against an observed default rate of "
                "{observed}."),
    "grades": ("Illustrative grades",
               "{n_grades} grades; grade {first} holds {share} of in-scope development loans."),
    "monitoring": ("Monitoring",
                   "Largest PSI {psi}: expected for a random split, not evidence of stability "
                   "over time."),
}
MODEL_CARD = {
    "Intended use": "learning and demonstration of PD modelling practice.",
    "Not for": "credit decisions.",
    "Limitations": "the model has no loan-to-value or debt-to-income information, the core drivers "
                   "of mortgage risk. Both were excluded because their missing values give the "
                   "outcome away (D-017), which likely limits the model's ranking power.",
}

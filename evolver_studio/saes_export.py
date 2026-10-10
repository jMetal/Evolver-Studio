"""The results of a validation study as the files SAES reads.

SAES (https://github.com/jMetal/SAES) analyzes the results of an experimental study from two CSV
files: the results, one row per independent run of an algorithm on a problem with the value of one
quality indicator (`Algorithm`, `Instance`, `MetricName`, `ExecutionId`, `MetricValue`), and the
metrics, which say whether each indicator is maximized (`MetricName`, `Maximize`). The study's runs
are laid out here as those files, so that SAES can be run on them as on any other study. SAES is
not needed to write them; the Wilcoxon pivot table is written by `latex_tables`.

Nothing here depends on Streamlit.
"""

import pandas as pd

from evolver_studio.catalogue import is_maximized
from evolver_studio.validation_stats import indicator_names

RESULTS_COLUMNS = ("Algorithm", "Instance", "MetricName", "ExecutionId", "MetricValue")
METRICS_COLUMNS = ("MetricName", "Maximize")


def saes_results(runs: pd.DataFrame) -> pd.DataFrame:
    """The runs of a study as SAES's results file.

    Args:
        runs: The runs table of the study (see `validation_stats`).

    Returns:
        One row per run and indicator, with the columns `RESULTS_COLUMNS`; the execution of a run
        is its number.
    """
    results = runs.melt(
        id_vars=["contender", "problem", "Run"],
        value_vars=indicator_names(runs),
        var_name="MetricName",
        value_name="MetricValue",
    ).rename(columns={"contender": "Algorithm", "problem": "Instance", "Run": "ExecutionId"})
    return results.dropna(subset=["MetricValue"])[list(RESULTS_COLUMNS)].reset_index(drop=True)


def saes_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """The indicators of a study as SAES's metrics file, with whether each one is maximized.

    Args:
        runs: The runs table of the study.

    Returns:
        One row per indicator, with the columns `METRICS_COLUMNS`.
    """
    return pd.DataFrame(
        {
            "MetricName": indicator_names(runs),
            "Maximize": [is_maximized(name) for name in indicator_names(runs)],
        },
        columns=list(METRICS_COLUMNS),
    )

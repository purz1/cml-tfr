import pandas as pd
import numpy as np
import scipy

from pandas.api.types import is_numeric_dtype as is_pandas_dtype_numeric
from pandas.api.types import is_string_dtype as is_pandas_dtype_string

from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Union, Set, Optional


from typing import Set


def wilcoxon_rank_sum_test(
    data: pd.DataFrame,
    predicted_variable: str,
    variables: Optional[Union[List[str], str]] = None,
    keep: bool = False,
    max_p_value: float = 0.05,
    adjust_p_values: bool = False,
) -> pd.DataFrame:
    """
    Perform the Wilcoxon rank-sum test between a binary target variable and numerical features.

    Parameters:
        data (pd.DataFrame): DataFrame containing the dataset.
        predicted_variable (str): Binary target variable (e.g., bool, str like "yes"/"no", or 0/1).
        variables (Optional[List[str] | str]): Variables to include or exclude (see `keep`).
        keep (bool): Whether to include (`True`) or exclude (`False`) the specified variables.
        max_p_value (float): Max p-value threshold for filtering results.
        adjust_p_values (bool): Apply multiple testing correction using FDR if True.

    Returns:
        pd.DataFrame: Summary table with columns:
                      ['covariate', 'wilcoxon-statistic', 'p-value', 'number of samples']
    """
    if isinstance(variables, str):
        variables = [variables]
    elif variables is None:
        variables = data.drop(columns=predicted_variable).select_dtypes(include=['bool', 'object']).columns.tolist()

    # Select columns to use
    if keep:
        test_data = data[variables + [predicted_variable]].copy()
    else:
        test_data = data.drop(columns=variables).copy()

    # Normalize predicted variable into a boolean mask
    unique_vals = data[predicted_variable].dropna().unique()
    if len(unique_vals) != 2:
        raise ValueError(f"Expected binary variable with 2 unique values, got {len(unique_vals)}: {unique_vals}")

    val_true, val_false = unique_vals[0], unique_vals[1]
    test_data[predicted_variable] = test_data[predicted_variable] == val_true  # Convert to boolean mask

    results = {'covariate': [], 'wilcoxon-statistic': [], 'p-value': [], 'number of samples': []}

    for variable in test_data.columns:
        if variable == predicted_variable or not is_pandas_dtype_numeric(test_data[variable]):
            continue

        temp_data = test_data[[predicted_variable, variable]].dropna()
        if temp_data.empty:
            continue

        try:
            mask = temp_data[predicted_variable].to_numpy(dtype=bool)
            values = temp_data[variable].to_numpy()
            group1 = values[mask]
            group2 = values[~mask]

            result = scipy.stats.ranksums(group1, group2)
            results['covariate'].append(variable)
            results['wilcoxon-statistic'].append(result.statistic)
            results['p-value'].append(result.pvalue)
            results['number of samples'].append(len(temp_data))
        except Exception as e:
            print(f"Failed to test variable: {variable} ({e})")

    results_df = pd.DataFrame(results)

    if adjust_p_values and not results_df.empty:
        results_df['p-value'] = multipletests(results_df['p-value'], method='fdr_bh')[1]

    results_df = results_df.sort_values('p-value').round(3)
    return results_df[results_df['p-value'] <= max_p_value]


def wilcoxon_rank_sum_test_with_collinearity_filter(
    df: pd.DataFrame,
    predicted_variable: str,
    sample_statistic_variables: List[str],
    image_variables: List[str],
    variables: Optional[Union[List[str], str]] = None,
    keep: bool = False,
    correlation_threshold: float = 0.8,
    max_p_value: float = 0.05,
) -> pd.DataFrame:
    """
    Run Wilcoxon rank-sum test and filter multicollinear variables based on Spearman correlation.

    Parameters:
        df (pd.DataFrame): Dataset for analysis.
        predicted_variable (str): Binary outcome variable.
        sample_statistic_variables (List[str]): Variables considered "cell statistics".
        image_variables (List[str]): Variables considered "image-based".
        variables (Optional[List[str] | str]): Variables to include or exclude (see `keep`).
        keep (bool): Whether to include (`True`) or exclude (`False`) the specified variables.
        correlation_threshold (float): Threshold for multicollinearity removal.
        max_p_value (float): Max p-value to include in results.

    Returns:
        pd.DataFrame: Filtered Wilcoxon test results excluding highly correlated variables.
    """
    results = wilcoxon_rank_sum_test(
        data=df,
        predicted_variable=predicted_variable,
        variables=variables,
        keep=keep,
        max_p_value=max_p_value
    )

    results["cell_statistics_variable"] = results["covariate"].isin(sample_statistic_variables)
    variables_to_drop: Set[str] = set()

    for value in results["cell_statistics_variable"].unique():
        subset = results[results["cell_statistics_variable"] == value]
        covariates = subset["covariate"].tolist()
        corr_matrix = df[covariates].corr(method="spearman").abs()

        for i, var1 in enumerate(corr_matrix.columns):
            for j in range(i + 1, len(corr_matrix.columns)):
                var2 = corr_matrix.columns[j]
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    p1 = subset.loc[subset["covariate"] == var1, "p-value"].values[0]
                    p2 = subset.loc[subset["covariate"] == var2, "p-value"].values[0]

                    if p1 > p2 and var1 in image_variables:
                        variables_to_drop.add(var1)
                    elif var2 in image_variables:
                        variables_to_drop.add(var2)

    return results[~results["covariate"].isin(variables_to_drop)].sort_values(by="p-value")


def summarize_continuous_variables_by_group(
    df: pd.DataFrame,
    group_var: str,
    variables: List[str],
    labels: Dict[Union[bool, str, int], str] = {True: 'yes', False: 'no'},
    all_label: str = 'All patients',
    num_decimals: int = 1
) -> pd.DataFrame:
    """
    Summarize continuous variables (median [Q1–Q3]) grouped by a categorical variable,
    including overall statistics.

    Parameters:
        df (pd.DataFrame): Input dataset.
        group_var (str): Categorical variable to group by (e.g., relapse status).
        variables (List[str]): List of continuous variable names.
        labels (Dict[Union[bool, str, int], str]): Mapping of group values to labels.
        all_label (str): Label for the overall statistics column.
        num_decimals (int): Number of decimal places to round summary stats.

    Returns:
        pd.DataFrame: Summary table including sample sizes and group-wise medians and IQRs.
    """
    summary_rows = []

    for var in variables:
        row = {'covariate': var}

        non_missing = df[var].notna().sum()
        total = len(df)
        row['Number of samples'] = (
            f"{non_missing}/{total} ({100 * non_missing / total:.1f}%)"
            if non_missing else "No data available (n=0)"
        )

        for key, label in {None: all_label, **labels}.items():
            group = df if key is None else df[df[group_var] == key]
            values = group[var].dropna()
            if values.empty:
                row[label] = "No data available (n=0)"
            else:
                med = round(values.median(), num_decimals)
                q1 = round(values.quantile(0.25), num_decimals)
                q3 = round(values.quantile(0.75), num_decimals)
                row[label] = f"{med} [{q1}-{q3}]"

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def summarize_categorical_variables_by_group(
    df: pd.DataFrame,
    group_var: str,
    variables: List[str],
    labels: Dict[Union[bool, str, int], str] = {True: 'yes', False: 'no'},
    all_label: str = 'All patients',
    num_decimals: int = 1
) -> pd.DataFrame:
    """
    Summarize categorical variables by showing counts and percentages per group.

    Parameters:
        df (pd.DataFrame): Input dataset.
        group_var (str): Name of the categorical grouping variable (e.g., relapse status).
        variables (List[str]): List of categorical variable names to summarize.
        labels (Dict[Union[bool, str, int], str]): Mapping of group values to display labels.
        all_label (str): Label for the overall summary column.
        num_decimals (int): Number of decimal places for percentages.

    Returns:
        pd.DataFrame: Summary table showing frequency and percentage of each category by group.
    """
    summary_rows = []

    for var in variables:
        total = len(df)
        non_missing = df[var].notna().sum()
        sample_text = f"{non_missing}/{total} ({100 * non_missing / total:.{num_decimals}f}%)"

        for category in df[var].dropna().unique():
            row = {
                'covariate': var,
                'category': category,
                'Number of samples': sample_text
            }

            for key, label in {None: all_label, **labels}.items():
                group = df if key is None else df[df[group_var] == key]
                group_non_missing = group[var].notna().sum()
                count = (group[var] == category).sum()
                percent = 100 * count / group_non_missing if group_non_missing else 0
                row[label] = f"{count} ({percent:.{num_decimals}f}%)"

            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def chi_square_test(
    data: pd.DataFrame, 
    predicted_variable: str, 
    variables: List[str] = None, 
    max_p_value: float = 0.05,
):
    results = {'covariate': [], 'chi2': [], 'p-value': [], 'number of samples': []}
    
    if variables == None:
        # Extract variables that are not bool or str type
        variables = data.select_dtypes(include=['bool', 'object']).columns.tolist()
    
    for variable in variables:
        if variable != predicted_variable:
            
            temp_data = data[[predicted_variable, variable]].dropna()
            number_of_samples = temp_data.shape[0]
            temp_data = pd.crosstab(temp_data[predicted_variable], temp_data[variable])

            if number_of_samples != 0:
            
                try:
                    chi2, pvalue, _, _ = scipy.stats.chi2_contingency(temp_data)

                    results['covariate'].append(variable)
                    results['chi2'].append(chi2)
                    results['p-value'].append(pvalue)
                    results['number of samples'].append(number_of_samples)
                except:
                    print(f'Failed to test variable: {variable}')

    results = pd.DataFrame(results)
    results = results.sort_values(by='p-value', ascending=True).round(3)
    results = results[results['p-value'] <= max_p_value]

    return results
    
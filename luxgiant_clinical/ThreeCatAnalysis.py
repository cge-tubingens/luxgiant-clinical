"""
Module with functions tailored to get summary statistics with grouped data with three categories
"""

import os
import subprocess

import numpy as np
import pandas as pd

from scipy import stats
import luxgiant_clinical.TwoCatAnalysis as two

def summaryze_mean_std(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

    """
    Summarize mean and standard deviation (SD) statistics from a grouped DataFrame (`df_grouped`)
    into a summary DataFrame (`df_sum`).

    This function computes and populates the `df_sum` DataFrame with the mean and SD values
    for each variable (`variables`) across specified groups (`groups`). It assumes that `df_grouped`
    contains pre-aggregated statistics for each variable across groups.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The summary DataFrame where the results will be populated.
    df_grouped : pd.DataFrame
        The grouped DataFrame containing statistical measures (mean, std, count) for each variable.
        Must have columns 'Variable', 'Stat' (containing 'mean', 'std', 'count'), and columns for each group.
    variables : list
        List of variables (column names) for which mean and SD are summarized.
    groups : list
        List of column names representing the groups for which statistics are summarized.

    Returns
    -------
    pd.DataFrame
        A DataFrame (`df_sum`) with columns 'Variable', 'Statistical Measure', group columns specified in `groups`,
        'Available Samples for Analysis' (total count of samples).

    Notes
    -----
    - Assumes `df_grouped` has columns 'Variable', 'Stat' ('mean', 'std', 'count'), and group columns.
    - Adds rows to `df_sum` with 'mean (SD)' for each variable across groups.
    - Computes 'Available Samples for Analysis' as the sum of counts across all specified groups.

    """
    
    num_rows = df_sum.shape[0]

    group_1= groups[0]
    group_2= groups[1]
    group_3= groups[2]

    for k,col in enumerate(variables):

        mask_feat = (df_grouped['Variable']==col)
        mask_mean = (df_grouped['Stat']=='mean')
        mask_std  = (df_grouped['Stat']=='std')
        mask_coun = (df_grouped['Stat']=='count')

        mean = df_grouped.loc[mask_feat & mask_mean].reset_index(drop=True)
        std  = df_grouped.loc[mask_feat & mask_std].reset_index(drop=True)
        count= df_grouped.loc[mask_feat & mask_coun].reset_index(drop=True)

        df_sum.loc[num_rows+k,'Variable'] = col
        df_sum.loc[num_rows+k,'Statistical Measure'] = 'mean (SD)'
        df_sum.loc[num_rows+k,group_1] = f"{round(mean[group_1][0],1)} ({round(std[group_1][0],1)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(mean[group_2][0],1)} ({round(std[group_2][0],1)})"
        df_sum.loc[num_rows+k,group_3] = f"{round(mean[group_3][0],1)} ({round(std[group_3][0],1)})"
        df_sum.loc[num_rows+k,'Available Samples for Analysis'] = f"{count[group_1][0]+count[group_2][0]+count[group_3][0]}"

    return df_sum

def one_way_anova(df_data:pd.DataFrame, variables:list, grouping:str)->pd.DataFrame:

    """
    Perform one-way ANOVA (Analysis of Variance) for each variable (`variables`) in the DataFrame (`df_data`)
    across the groups defined by `grouping`.

    Parameters
    ----------
    df_data : pd.DataFrame
        The input DataFrame containing the data.
    variables : list
        List of column names (variables) for which ANOVA will be performed.
    grouping : str
        The column name representing the grouping variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of one-way ANOVA for each variable.
        Columns include 'Variable' (variable name) and 'p-value' (rounded to 4 decimal places).

    Notes
    -----
    - Drops rows with NaN values in the specified variables before performing ANOVA.
    - Uses scipy.stats.f_oneway to compute ANOVA statistics.
    - Assumes `df_data` contains columns specified in `variables` and `grouping`.

    """

    results = []
    
    # Get unique groups
    groups = df_data[grouping].unique()
    
    for var in variables:

        df_nonull = df_data.dropna(subset=var, inplace=False)

        samples = [df_nonull[df_nonull[grouping] == group][var] for group in groups]

        stat, p_value = stats.f_oneway(*samples)
        results.append({'Variable': var, 'p-value': round(p_value, 4)})
        
    results_df = pd.DataFrame(results)

    return results_df

def summaryze_count_percent(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

    """
    Summarizes count and percentage statistics from a grouped DataFrame (`df_grouped`) into a summary DataFrame (`df_sum`).

    Parameters
    ----------
    df_sum : pd.DataFrame
        The DataFrame to store the summarized statistics.
    df_grouped : pd.DataFrame
        The grouped DataFrame containing statistics to be summarized.
        Must have columns 'Variable', 'Stat', and columns corresponding to groups in `groups`.
    variables : list
        List of variables (column names) to summarize from `df_grouped`.
    groups : list
        List of group names (column names) in `df_grouped` where statistics are summarized across.

    Returns
    -------
    pd.DataFrame
        A DataFrame (`df_sum`) containing summarized count and percentage statistics for each variable across groups.
        Columns include 'Variable', 'Statistical Measure', group names from `groups`, and 'Available Samples for Analysis'.

    Notes
    -----
    - Assumes `df_grouped` has columns 'Variable', 'Stat', and columns corresponding to `groups`.
    - Updates `df_sum` with new rows for each variable in `variables`, summarizing count and percentage statistics across `groups`.

    """
    
    num_rows = df_sum.shape[0]

    group_1 = groups[0]
    group_2 = groups[1]
    group_3 = groups[2]

    for k,col in enumerate(variables):

        mask_feat = (df_grouped['Variable']==col)
        mask_sum  = (df_grouped['Stat']=='sum')
        mask_perc = (df_grouped['Stat']=='%')
        mask_coun = (df_grouped['Stat']=='count')

        sum  = df_grouped.loc[mask_feat & mask_sum].reset_index(drop=True)
        perc = df_grouped.loc[mask_feat & mask_perc].reset_index(drop=True)
        count= df_grouped.loc[mask_feat & mask_coun].reset_index(drop=True)

        df_sum.loc[num_rows+k,'Variable'] = col
        df_sum.loc[num_rows+k,'Statistical Measure'] = 'n (%)'
        df_sum.loc[num_rows+k,group_1] = f"{round(sum[group_1][0])} ({round(perc[group_1][0],1)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(sum[group_2][0])} ({round(perc[group_2][0],1)})"
        df_sum.loc[num_rows+k,group_3] = f"{round(sum[group_3][0])} ({round(perc[group_3][0],1)})"
        df_sum.loc[num_rows+k,'Available Samples for Analysis'] = f"{count[group_1][0]+count[group_2][0]+count[group_3][0]}"
    
    return df_sum

def decide_fisher_chi2(feature_1:pd.Series, feature_2:pd.Series)->bool:

    """
    Decide whether to use Fisher's exact test or chi-square test based on expected cell counts.

    Parameters
    ----------
    feature_1 : pd.Series
        First categorical feature (variable) for contingency table analysis.
    feature_2 : pd.Series
        Second categorical feature (variable) for contingency table analysis.

    Returns
    -------
    bool
        True if Fisher's exact test should be used (expected cell count < 5 in more than 20% of cells), 
        False if chi-square test can be used.

    Notes
    -----
    - Computes a contingency table (`df_cross`) using `pd.crosstab` to count occurrences of feature pairs.
    - Calculates expected cell counts and determines if more than 20% of expected counts are below 5.
    - Returns True if Fisher's exact test is recommended, False if chi-square test can be used.

    """

    df_cross = pd.crosstab(feature_1, feature_2, margins=True)

    counter = []

    for k in range(len(df_cross)-1):
        for m in range(len(df_cross)-1):
            expected = (df_cross.iloc[-1,m]*df_cross.iloc[k,-1])/df_cross.iloc[-1,-1]
            if expected <5:
                counter.append(1)
            else:
                counter.append(0)

    percent = sum(counter)/len(counter)
    if percent>0.2: return True # requires Fisher's exact test
    else: return False          # chi2 test can be used

def chi2_fisher_exact(feature_1:pd.Series, feature_2:pd.Series)->float:

    """
    Perform Fisher's exact test or chi-square test based on expected cell counts and return the p-value.

    Parameters
    ----------
    feature_1 : pd.Series
        First categorical feature (variable) for contingency table analysis.
    feature_2 : pd.Series
        Second categorical feature (variable) for contingency table analysis.

    Returns
    -------
    float
        The p-value resulting from Fisher's exact test or chi-square test.

    Notes
    -----
    - Computes a contingency table (`crosstab`) using `pd.crosstab` to count occurrences of feature pairs.
    - Saves the contingency table to a CSV file (`crosstab.csv`) in the current script's directory.
    - Determines whether to use Fisher's exact test or chi-square test using `decide_fisher_chi2`.
    - If Fisher's exact test is chosen:
        - Converts the contingency table to CSV format and runs an R script (`run_r_fisher_test`) to compute the p-value.
        - Deletes the temporary CSV files (`input_csv` and `output_csv`) after computation.
    - If chi-square test is chosen:
        - Uses `stats.chi2_contingency` from scipy.stats to compute the p-value.

    """

    crosstab = pd.crosstab(feature_1, feature_2)

    script_dir = os.path.dirname(__file__)

    temp = pd.DataFrame(crosstab.values, columns=crosstab.columns.to_list())

    temp.to_csv(os.path.join(script_dir, 'crosstab.csv'), index=False)

    if decide_fisher_chi2(feature_1, feature_2):

        input_csv = os.path.join(script_dir, 'crosstab.csv')
        output_csv= os.path.join(script_dir, 'result.csv')

        fisher_p_val = run_r_fisher_test(input_csv, output_csv).iloc[0,1]

        if os.path.exists(output_csv): os.remove(output_csv)

        if os.path.exists(input_csv): os.remove(input_csv)

        return fisher_p_val
    else:
        return stats.chi2_contingency(crosstab, correction=False).pvalue
    
def run_r_fisher_test(input_csv, output_csv):

    """
    Run an R script to perform Fisher's exact test on a contingency table and return the results.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file containing the contingency table data.
    output_csv : str
        Path to the output CSV file where the results of the Fisher's exact test will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the Fisher's exact test.

    Notes
    -----
    - Uses an external R script (`exactFisher.R`) to compute Fisher's exact test.
    - Reads the contingency table data from `input_csv` and saves the results to `output_csv`.
    - The R script should take `input_csv` and `output_csv` as command-line arguments.
    - Uses `subprocess.run` to execute the R script and waits for it to complete.
    - Reads and returns the results from the `output_csv` file as a pandas DataFrame.

    """

    # Get the path to the R script
    script_dir = os.path.dirname(__file__)
    r_script_path = os.path.join(script_dir, "exactFisher.R")
    
    # Define the command to run the R script
    command = ["Rscript", r_script_path, input_csv, output_csv]
    
    # Run the command
    subprocess.run(command, check=True)

    # Read and return the results from the output CSV
    result = pd.read_csv(output_csv)
    return result

def chi_squared_tests(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    """
    Perform chi-squared or Fisher's exact tests on contingency tables for each variable.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame containing the data.
    variables : list
        List of column names (features) in `df_data` for which tests will be performed.
    group_var : str
        Column name in `df_data` representing the grouping variable for contingency tables.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of chi-squared or Fisher's exact tests for each variable.
        Columns include 'Variable' (name of the variable) and 'p-value' (resulting p-value from the test).

    Notes
    -----
    - Uses the `chi2_fisher_exact` function to determine whether to perform Fisher's exact test
      or chi-squared test based on the requirements (using Fisher's exact test if cell counts are low).
    - Constructs contingency tables for each variable and calculates p-values.
    - Returns results in a DataFrame format with columns 'Variable' and 'p-value'.

    """

    crosstab_results = {}
    for var in variables:

        pval = chi2_fisher_exact(df_data[var], df_data[group_var])
        crosstab_results[var] = {'p_val': pval}

    result = pd.DataFrame(crosstab_results).transpose()
    result = result.reset_index()

    result.columns = ['Variable', 'p-value']

    return result

def summaryze_median_iqr(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

    """
    Summarizes median and interquartile range (IQR) statistics for each variable across multiple groups.

    Parameters
    ----------
    df_sum : pd.DataFrame
        DataFrame to store summarized results.
    df_grouped : pd.DataFrame
        DataFrame containing grouped statistics.
    variables : list
        List of column names (features) in `df_grouped` to summarize.
    groups : list
        List of column names in `df_grouped` representing groups for statistical summaries.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame `df_sum` with summarized statistics for each variable across groups.
        Columns include 'Variable', 'Statistical Measure', groups, and 'Available Samples for Analysis'.

    Notes
    -----
    - Constructs summary statistics (median, first quartile, third quartile, count) for each variable
      across groups specified in `groups`.
    - Assumes `df_grouped` contains columns 'Variable', 'Stat' (statistics type), and group columns.
    - Updates `df_sum` with rows corresponding to each variable summarizing median and IQR values
      for each group.
    - 'Statistical Measure' is set to 'median (IQR)' for each variable.
    - 'Available Samples for Analysis' sums the counts across all groups for each variable.

    """
    
    num_rows = df_sum.shape[0]

    group_1 = groups[0]
    group_2 = groups[1]
    group_3 = groups[2]

    for k,col in enumerate(variables):

        mask_feat = (df_grouped['Variable']==col)
        mask_medn = (df_grouped['Stat']=='median')
        mask_fstq = (df_grouped['Stat']=='first_Q')
        mask_trdq = (df_grouped['Stat']=='third_Q')
        mask_coun = (df_grouped['Stat']=='count')

        medn = df_grouped.loc[mask_feat & mask_medn].reset_index(drop=True)
        fstq = df_grouped.loc[mask_feat & mask_fstq].reset_index(drop=True)
        trdq = df_grouped.loc[mask_feat & mask_trdq].reset_index(drop=True)
        count= df_grouped.loc[mask_feat & mask_coun].reset_index(drop=True)

        df_sum.loc[num_rows+k,'Variable'] = col
        df_sum.loc[num_rows+k,'Statistical Measure'] = 'median (IQR)'
        df_sum.loc[num_rows+k,group_1] = f"{round(medn[group_1][0],1)} ({round(fstq[group_1][0],1)} - {round(trdq[group_1][0],1)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(medn[group_2][0],1)} ({round(fstq[group_2][0],1)} - {round(trdq[group_2][0],1)})"
        df_sum.loc[num_rows+k,group_3] = f"{round(medn[group_3][0],1)} ({round(fstq[group_3][0],1)} - {round(trdq[group_3][0],1)})"
        df_sum.loc[num_rows+k,'Available Samples for Analysis'] = f"{count[group_1][0]+count[group_2][0]+count[group_3][0]}"

    return df_sum

def kruskal_wallis_test(data:pd.DataFrame, grouping:str, test_cols:list)->pd.DataFrame:

    """
    Perform Kruskal-Wallis test to compare medians of multiple groups for each variable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    grouping : str
        Column name in `data` representing groups for comparison.
    test_cols : list
        List of column names (variables) in `data` to test for differences across groups.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Variable' and 'p-value' containing results of Kruskal-Wallis test for each variable.

    Raises
    ------
    ValueError
        If the number of unique groups in `data[grouping]` is not equal to 3.

    Notes
    -----
    - Kruskal-Wallis test is a non-parametric test to determine if there are differences
      between the medians of three or more independent groups.
    - `nan_policy='omit'` is used to handle NaN values by omitting them during the test.
    - Results DataFrame has rows corresponding to each variable in `test_cols` with columns
      'Variable' (name of the variable) and 'p-value' (significance level).
    - Raises an error if the number of unique groups in `data[grouping]` is not exactly 3,
      as Kruskal-Wallis test requires exactly three groups.
    """

    results = []
    
    # Get unique groups
    groups = data[grouping].unique()
    if len(groups) != 3:
        raise ValueError("There must be exactly three groups for this test.")
    
    # Perform the Kruskal-Wallis test for each variable
    for col in test_cols:
        samples = [data[data[grouping] == group][col] for group in groups]
        stat, p_value = stats.kruskal(*samples, nan_policy='omit')
        results.append({'Variable': col, 'p-value': round(p_value, 4)})
    
    results_df = pd.DataFrame(results)

    return results_df

def report_mean_std(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, subheader:str=None)->pd.DataFrame:

    """
    Generate a summary report of mean, standard deviation, and hypothesis testing for a list of variables, 
    grouped by a specified column.

    Parameters:
    -----------
    data_df (pd.DataFrame): 
        The input DataFrame containing the data.
    variables (list): 
        A list of column names (strings) in data_df for which the summary statistics and hypothesis testing 
        will be performed.
    groups (list): 
        A list of unique values in the grouping_by column to be used as groups.
    grouping_by (str): 
        The name of the column in data_df by which to group the data.

    Returns:
    --------
    pd.DataFrame: A DataFrame with summary statistics (mean and standard deviation) for each variable, 
                  both grouped by the specified column and overall.
                  The DataFrame also includes results of hypothesis testing (ANOVA) and the number of 
                  available samples for analysis.

    Raises:
    ------
    KeyError: If any of the variables in the variables list or the grouping_by column are not present in data_df.
    """

    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.mean_std(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = summaryze_mean_std(
            df_summary, 
            stats_by_group, 
            variables, 
            groups
        )
    
    # merge results
    df_summary = df_summary\
        .merge(
            two.mean_std_simple(data_df, features=variables) # descriptive statistics for ungrouped data
        )\
        .merge(
            one_way_anova(data_df, variables, grouping_by), on='Variable' # hypothesis testing
        )
    
    ordered_cols = summary_cols[:-1] + ['Total', 'p-value', 'Available Samples for Analysis']

    if subheader is not None:
        values = [subheader] + [np.nan]*(len(ordered_cols)-1)
        subhead = pd.DataFrame(data=[values], columns=ordered_cols)

        df_summary = pd.concat(
            [subhead, df_summary[ordered_cols].copy()], axis=0, ignore_index=True
        )
        return df_summary
    else: 
        return df_summary[ordered_cols].copy()

def report_median_iqr(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, subheader:str=None)->pd.DataFrame:

    """
    Generate a summary report of median, interquartile range (IQR), and hypothesis testing for a list of variables, 
    grouped by a specified column.

    Parameters:
    -----------
    data_df (pd.DataFrame): 
        The input DataFrame containing the data.
    variables (list): 
        A list of column names (strings) in data_df for which the summary statistics and hypothesis testing will be performed.
    groups (list): 
        A list of unique values in the grouping_by column to be used as groups.
    grouping_by (str): 
        The name of the column in data_df by which to group the data.

    Returns:
    --------
    pd.DataFrame: A DataFrame with summary statistics (median and IQR) for each variable, both grouped by 
                  the specified column and overall.
                  The DataFrame also includes results of hypothesis testing (Kruskal-Wallis test) and 
                  the number of available samples for analysis.

    Raises:
    ------
    KeyError: If any of the variables in the variables list or the grouping_by column are not present in data_df.
    """

    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.median_iqr(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = summaryze_median_iqr(df_summary, stats_by_group, variables, groups)

    # merge results
    df_summary = df_summary\
        .merge(
            two.median_iqr_simple(data_df, variables), on='Variable'
        )\
        .merge(
            kruskal_wallis_test(data_df, grouping_by, test_cols=variables), on='Variable'
        )

    ordered_cols = summary_cols[:-1] + ['Total', 'p-value', 'Available Samples for Analysis']

    if subheader is not None:
        values = [subheader] + [np.nan]*(len(ordered_cols)-1)
        subhead = pd.DataFrame(data=[values], columns=ordered_cols)

        df_summary = pd.concat(
            [subhead, df_summary[ordered_cols].copy()], axis=0, ignore_index=True
        )
        return df_summary
    else: 
        return df_summary[ordered_cols].copy()

def report_proportion(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, subheader:str=None)->pd.DataFrame:

    """
    Generate a report summarizing proportions and chi-squared test results for categorical variables.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data.
    variables : list
        List of column names (variables) in `data_df` to analyze.
    groups : list
        List of group names to compare (typically two groups).
    grouping_by : str
        Column name in `data_df` used for grouping the analysis.
    subheader : str, optional
        Subheader to prepend to the report, default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing proportions, chi-squared test results, and other statistics
        for each variable across specified groups.

    Notes
    -----
    - Uses `two.count_percent` and `two.count_simple` functions to calculate proportions and counts.
    - Uses `chi_squared_tests` function to perform chi-squared tests for categorical associations.
    - Resulting DataFrame includes columns 'Variable', 'Statistical Measure', group columns,
      'Total', 'p-value', and 'Available Samples for Analysis'.
    - If `subheader` is provided, it prepends a row with this value to the DataFrame.
    """
    
    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.count_percent(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = summaryze_count_percent(df_summary, stats_by_group, variables, groups)

    # merge results
    df_summary = df_summary\
        .merge(
            two.count_simple(data_df, features=variables), on='Variable'
        )\
        .merge(
            chi_squared_tests(data_df, variables, grouping_by), on='Variable'
        )
    
    ordered_cols = summary_cols[:-1] + ['Total', 'p-value', 'Available Samples for Analysis']

    if subheader is not None:
        values = [subheader] + [np.nan]*(len(ordered_cols)-1)
        subhead = pd.DataFrame(data=[values], columns=ordered_cols)

        df_summary = pd.concat(
            [subhead, df_summary[ordered_cols].copy()], axis=0, ignore_index=True
        )
        return df_summary
    else: 
        return df_summary#[ordered_cols].copy()
    
def bonferroni_mean_std(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, correc_factor:int)->pd.DataFrame:

    """
    Perform Bonferroni correction on p-values obtained from mean and standard deviation analysis.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data.
    variables : list of str
        List of column names (variables) in `data_df` to analyze.
    groups : list of str
        List of group names to compare.
    grouping_by : str
        Column name in `data_df` used for grouping the analysis.
    correc_factor : int
        Bonferroni correction factor to adjust p-values.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted p-values after Bonferroni correction for each variable.

    Notes
    -----
    - Uses `two.mean_std` to calculate mean and standard deviation statistics.
    - Uses `two.summaryze_mean_std` to summarize mean and standard deviation statistics.
    - Uses `two.mean_std_simple` to obtain simple mean and standard deviation values.
    - Uses `two.t_test_by_group` to perform t-tests by group for statistical significance.
    - Resulting DataFrame includes columns 'Variable' and 'Adjusted p-value'.
    - Adjusts p-values by multiplying the original p-values by the Bonferroni correction factor.
    - Renames columns for clarity.
    """
    
    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.mean_std(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = two.summaryze_mean_std(df_summary, stats_by_group, variables, groups)

    # merge results
    df_summary = df_summary\
        .merge(
            two.mean_std_simple(data_df, variables)
        )\
        .merge(
            two.t_test_by_group(data_df, variables, grouping_by), on='Variable'
        )
    
    # keep only needed columns
    df_summary = df_summary[['Variable', 'p-value']].copy()

    # Bonferroni adjustment
    df_summary['p-value'] = correc_factor*df_summary['p-value']

    # rename columns
    df_summary.columns = ['Variable', 'Adjusted p-value']
    
    return df_summary

def bonferroni_median_iqr(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, correc_factor:int)->pd.DataFrame:

    """
    Perform Bonferroni correction on p-values obtained from median and IQR analysis.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data.
    variables : list of str
        List of column names (variables) in `data_df` to analyze.
    groups : list of str
        List of group names to compare.
    grouping_by : str
        Column name in `data_df` used for grouping the analysis.
    correc_factor : int
        Bonferroni correction factor to adjust p-values.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted p-values after Bonferroni correction for each variable.

    Notes
    -----
    - Uses `two.median_iqr` to calculate median and interquartile range (IQR) statistics.
    - Uses `two.summaryze_median_iqr` to summarize median and IQR statistics.
    - Uses `two.median_iqr_simple` to obtain simple median and IQR values.
    - Uses `two.mann_whitney` to perform Mann-Whitney U test by group for statistical significance.
    - Resulting DataFrame includes columns 'Variable' and 'Adjusted p-value'.
    - Adjusts p-values by multiplying the original p-values by the Bonferroni correction factor.
    - Renames columns for clarity.
    """
    
    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.median_iqr(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = two.summaryze_median_iqr(df_summary, stats_by_group, variables, groups)

    # merge results
    df_summary = df_summary\
        .merge(
            two.median_iqr_simple(data_df, variables), on='Variable'
        )\
        .merge(
            two.mann_whitney(data_df, variables, grouping_by), on='Variable'
        )
    
    # keep only needed columns
    df_summary = df_summary[['Variable', 'p-value']].copy()

    # Bonferroni adjustment
    df_summary['p-value'] = correc_factor*df_summary['p-value']

    # rename columns
    df_summary.columns = ['Variable', 'Adjusted p-value']

    return df_summary

def bonferroni_proportions(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, correc_factor:int, subheader:str=None)->pd.DataFrame:

    """
    Perform Bonferroni correction on p-values obtained from proportion analysis.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data.
    variables : list of str
        List of column names (variables) in `data_df` to analyze.
    groups : list of str
        List of group names to compare.
    grouping_by : str
        Column name in `data_df` used for grouping the analysis.
    correc_factor : int
        Bonferroni correction factor to adjust p-values.
    subheader : str, optional
        Optional subheader to prepend to the results DataFrame, default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted p-values after Bonferroni correction for each variable.

    Notes
    -----
    - Uses `two.count_percent` to calculate counts and percentages within each group.
    - Uses `two.summaryze_count_percent` to summarize counts and percentages.
    - Uses `two.count_simple` to obtain simple count and percentage values.
    - Uses `two.chi_squared_tests` to perform Chi-squared tests of independence by group for statistical significance.
    - Resulting DataFrame includes columns 'Variable' and 'Adjusted p-value'.
    - Adjusts p-values by multiplying the original p-values by the Bonferroni correction factor.
    - Renames columns for clarity.
    - Optionally includes a subheader row at the beginning of the DataFrame if `subheader` is provided.

    """
    
    # create empty dataframe to store results
    summary_cols = ['Variable', 'Statistical Measure'] + groups + ['Available Samples for Analysis']
    df_summary = pd.DataFrame(columns=summary_cols)

    # descriptive statistics by groups
    stats_by_group = two.count_percent(data_df, variables, grouping_by)

    # format table with descriptive statistics
    df_summary = two.summaryze_count_percent(df_summary, stats_by_group, variables, groups)

    # merge results
    df_summary = df_summary\
        .merge(
            two.count_simple(data_df, variables), on='Variable'
        )\
        .merge(
            two.chi_squared_tests(data_df, variables, grouping_by), on='Variable'
        )
    
    # keep only needed columns
    df_summary = df_summary[['Variable', 'p-value']].copy()

    # Bonferroni adjustment
    df_summary['p-value'] = correc_factor*df_summary['p-value']

    # rename columns
    df_summary.columns = ['Variable', 'Adjusted p-value']

    if subheader is not None:
        values = [subheader, np.nan]
        subhead = pd.DataFrame(data=[values], columns=['Variable', 'Adjusted p-value'])

        df_summary = pd.concat(
            [subhead, df_summary], axis=0, ignore_index=True
        )
        return df_summary
    else:
        return df_summary
    
def final_formatter(overall_df:pd.DataFrame, adjusted_df:list, groups:list)->pd.DataFrame:

    """
    Format and merge multiple DataFrames containing statistical results for final presentation.

    Parameters
    ----------
    overall_df : pd.DataFrame
        DataFrame containing overall statistical results.
    adjusted_df : list of pd.DataFrame
        List of DataFrames containing adjusted statistical results.
    groups : list of str
        List of group names used for statistical comparisons.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing merged statistical results.

    Notes
    -----
    - Uses a nested function `pvalue_formatter` to format p-values into readable strings.
    - Merges `overall_df` with each DataFrame in `adjusted_df` on 'Variable'.
    - Adds columns from each DataFrame in `adjusted_df` to `overall_df` based on 'Variable'.
    - Adjusted columns are identified by their column names containing 'p-value' or other statistical measures.
    - Includes columns 'Variable', 'Statistical Measure', group columns (`groups`), 'Total', 'p-value',
      and 'Available Samples for Analysis' in the final DataFrame.
    - Converts p-values in the DataFrame using `pvalue_formatter`.
    - Fills missing values in the DataFrame with empty strings.

    """

    orderded_cols = ['Variable', 'Statistical Measure'] + groups \
        +['Total', 'p-value']
    
    df = overall_df.copy()

    for data_df in adjusted_df:
        df = pd.merge(df, data_df, on='Variable')
        orderded_cols.append(data_df.columns[1])

    orderded_cols.append('Available Samples for Analysis')
    
    df = df[orderded_cols].copy()

    for col in df.columns:

        if 'p-value' in col:
            df[col] = df[col].apply(lambda x: pvalue_formatter(x))

    return df.fillna('')

def pvalue_formatter(p_val:float)->str:

    """
    Format p-values into readable strings.

    Parameters
    ----------
    p_val : float
        p-value to be formatted.

    Returns
    -------
    str
        Formatted string representation of the p-value.

    Notes
    -----
    - Returns an empty string if p_val is NaN.
    - Returns '0.9999' if p_val >= 1.
    - Returns 'p<0.001' if p_val < 0.001.
    - Otherwise, returns the p_val rounded to four decimal places as a string.
    """

    from math import isnan

    if isnan(p_val): return ''
    elif p_val>=0.001 and p_val<1: return str(round(p_val,4))
    elif p_val>=1: return '0.9999'
    else: return 'p<0.001'

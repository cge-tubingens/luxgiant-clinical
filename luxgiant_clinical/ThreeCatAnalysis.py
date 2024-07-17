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

    crosstab_results = {}
    for var in variables:

        pval = chi2_fisher_exact(df_data[var], df_data[group_var])
        crosstab_results[var] = {'p_val': pval}

    result = pd.DataFrame(crosstab_results).transpose()
    result = result.reset_index()

    result.columns = ['Variable', 'p-value']

    return result

def summaryze_median_iqr(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

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

def report_mean_std(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str)->pd.DataFrame:

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

    return df_summary[ordered_cols].copy()

def report_median_iqr(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str)->pd.DataFrame:

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

    return df_summary[ordered_cols].copy()

def report_proportion(data_df:pd.DataFrame, variables:list, groups:list, grouping_by:str, subheader:str=None)->pd.DataFrame:

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

    def pvalue_formatter(p_val:float)->str:

        from math import isnan

        if isnan(p_val): return ''
        elif p_val>=0.001 and p_val<1: return str(round(p_val,4))
        elif p_val>=1: return '0.9999'
        else: return 'p<0.001'

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

"""
Module with functions tailored to get summary statistics with grouped data with two categories
"""

import numpy as np
import pandas as pd

from scipy import stats

def decide_student_welch(group_1:pd.Series, group_2:pd.Series)->bool:

    """
    Decide whether to use Welch's t-test or Student's t-test based on Levene's test for equal variances.

    This function performs Levene's test to check the equality of variances between two groups.
    If the p-value from Levene's test is less than 0.05, it indicates that the variances are unequal,
    and therefore Welch's t-test should be used. Otherwise, Student's t-test should be used.

    Parameters
    ----------
    group_1 : pd.Series
        The first group of data as a Pandas Series.
    group_2 : pd.Series
        The second group of data as a Pandas Series.

    Returns
    -------
    bool
        True if Welch's t-test should be used (unequal variances), False if Student's t-test should be used 
        (equal variances).
    """

    # Levene Test's p-value 
    pval = stats.levene(group_1, group_2, nan_policy='omit').pvalue

    if pval < 0.05: return True # unequal variance => Welch t-test
    else: return False          # => Student t-test

def ttest_equal_unequal(group_1:pd.Series, group_2:pd.Series)->float:

    """
    Perform a t-test to compare the means of two groups, using either Welch's t-test or Student's t-test 
    based on the equality of variances.

    This function first checks whether the variances of the two groups are equal using the 
    `decide_student_welch` function. It then performs the appropriate t-test:
    - Welch's t-test if the variances are unequal
    - Student's t-test if the variances are equal

    Parameters
    ----------
    group_1 : pd.Series
        The first group of data as a Pandas Series.
    group_2 : pd.Series
        The second group of data as a Pandas Series.

    Returns
    -------
    float
        The p-value from the t-test indicating the probability of observing the data under the null hypothesis.
    """

    if decide_student_welch(group_1, group_2):
        return stats.ttest_ind(group_1.dropna(), group_2.dropna(), equal_var=False).pvalue
    else:
        return stats.ttest_ind(group_1.dropna(), group_2.dropna(), equal_var=True).pvalue

def t_test_by_group(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    """
    Perform t-tests for multiple variables between two groups in a DataFrame.

    This function computes the p-values for t-tests comparing the means of the specified variables
    between two groups defined by `group_var`. It automatically decides between Welch's t-test 
    and Student's t-test based on the equality of variances for each variable.

    Parameters
    ----------
    df_data : pd.DataFrame
        The input DataFrame containing the data.
    variables : list
        A list of variable names (columns) to be tested.
    group_var : str
        The name of the column used to define the two groups for comparison.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the variables and their corresponding p-values from the t-tests.

    Raises
    ------
    ValueError
        If the `group_var` column does not contain exactly two unique values.
    KeyError
        If any of the specified variables or `group_var` is not found in the DataFrame.
    """

    ttest_results = {}
    groups = df_data[group_var].unique().tolist()

    if len(groups) != 2:
        raise ValueError("The group_var column must contain exactly two unique values.")

    for var in variables:

        if var not in df_data.columns or group_var not in df_data.columns:
            raise KeyError("The specified variable or group_var is not in the DataFrame.")
        
        group1 = df_data[df_data[group_var] == groups[0]][var]
        group2 = df_data[df_data[group_var] == groups[1]][var]

        p_val = ttest_equal_unequal(group1, group2)
        ttest_results[var] = {'p_value': p_val}

    results = pd.DataFrame(ttest_results).transpose()

    results = results.reset_index()
    results.columns = ['Variable', 'p-value']

    return results

def mean_std_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    """
    Calculate the mean and standard deviation for a list of features in a DataFrame.

    Parameters:
    -----------
    data (pd.DataFrame): 
        The input DataFrame containing the data.
    features (list): 
        A list of column names (strings) in data for which the mean and standard deviation will be calculated.

    Returns:
    --------
    pd.DataFrame: 
        A DataFrame with the calculated mean and standard deviation for each feature. The DataFrame has 
        columns 'Variable' and 'Total'. 'Variable' contains the names of the features, and 'Total' contains 
        the mean and standard deviation formatted as "mean (std)".

    Raises:
    -------
    KeyError: If any of the features in the features list are not present in data.
    """

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        if feat not in data.columns:
            raise KeyError(f"The specified feature '{feat}' is not in the DataFrame.")

        mean= round(np.mean(data[feat]),1)
        std = round(np.std(data[feat]), 1)

        result.loc[feat, 'Total'] = f"{mean} ({std})"

    result = result.reset_index()

    result.columns = ['Variable', 'Total']

    return result

def mean_std(data:pd.DataFrame, features:list, grouping_by:str)->pd.DataFrame:

    """
    Calculate the mean, standard deviation, and count for a list of features in a DataFrame, 
    grouped by a specified column.

    Parameters:
    ----------
    data (pd.DataFrame): 
        The input DataFrame containing the data.
    features (list): 
        A list of column names (strings) in data for which the mean, standard deviation, 
        and count will be calculated.
    grouping_by (str): 
        The name of the column in data by which to group the data.

    Returns:
    --------
    pd.DataFrame: 
        A DataFrame with the calculated mean, standard deviation, and count for each feature, 
        grouped by the specified column. The DataFrame has columns 'Variable', 'Stat', and one 
        column for each unique value in the grouping_by column.

    Raises:
    ------
    KeyError: If any of the features in the features list or the grouping_by column are not present in data.
    """

    for feat in features:
        if feat not in data.columns:
            raise KeyError(f"The specified feature '{feat}' is not in the DataFrame.")

    agg_dict = {feat: ['mean', 'std', 'count'] for feat in features}

    grouped = data.groupby(by=grouping_by, as_index=False)[features]\
        .agg(agg_dict)\
        .transpose()\
        .reset_index(drop=False)
    
    grouped.columns = grouped.loc[0,:].to_list()
    grouped.columns = ['Variable', 'Stat'] + list(grouped.columns[2:])
    grouped = grouped.loc[1:,:].copy()

    return grouped

def summaryze_mean_std(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

    """
    Summarize the mean and standard deviation for specified variables between two groups.

    This function updates `df_sum` with the mean and standard deviation for the specified variables 
    from `df_grouped`. The summary includes the mean, standard deviation (SD), and the total count 
    of available samples for analysis for each group.

    Parameters
    ----------
    df_sum : pd.DataFrame
        The DataFrame to update with the summarized statistics.
    df_grouped : pd.DataFrame
        The DataFrame containing the grouped statistics, with columns for 'Variable', 'Stat', and the groups.
    variables : list
        A list of variable names (columns) to summarize.
    groups : list
        A list of two group names to compare.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame `df_sum` with the summarized statistics.
    """

    num_rows = df_sum.shape[0]

    group_1 = groups[0]
    group_2 = groups[1]

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
        df_sum.loc[num_rows+k,'Available Samples for Analysis'] = f"{count[group_1][0]+count[group_2][0]}"

    return df_sum

def count_percent(data:pd.DataFrame, features:list, grouping_by:str)->pd.DataFrame:

    """
    Compute count and percentage for specified features grouped by a variable in a DataFrame.

    This function calculates the count and percentage of occurrences for each feature within 
    each group defined by `grouping_by` in the input DataFrame `data`.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    features : list
        A list of column names (features) in `data` for which to compute counts and percentages.
    grouping_by : str
        The name of the column in `data` by which to group and compute statistics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'Variable', 'Stat', and percentages ('%') for each feature 
        within each group defined by `grouping_by`.
    """

    agg_dict = {feat: ['sum', 'count'] for feat in features}

    grouped = data.groupby(by=grouping_by, as_index=False)[features]\
        .agg(agg_dict)\
        .transpose()\
        .reset_index(drop=False)
    
    grouped.columns = grouped.loc[0,:].to_list()
    grouped.columns = ['Variable', 'Stat'] + list(grouped.columns[2:])
    result = grouped.loc[1:,:].copy()

    result = grouped.transpose()

    multi_level = pd.MultiIndex.from_arrays([result.iloc[0], result.iloc[1]], names=['level_1', 'level_2'])
    result.columns = multi_level
    result = result.drop(['Variable', 'Stat']).reset_index(drop=True)

    for feat in features:
        result[(feat, '%')] = 100*(np.float64(result[(feat, 'sum')]))/np.float64(result[(feat, 'count')])

    result = result.transpose().reset_index()
    result.columns = result.iloc[0]
    result.columns = ['Variable', 'Stat'] + list(result.columns[2:])
    result = result.drop([0])

    return result

def decide_fisher_chi2(feature_1:pd.Series, feature_2:pd.Series)->bool:

    """
    Decide between Fisher's Exact Test and Chi-square Test based on expected frequencies.

    This function computes the expected frequencies using a contingency table (crosstab)
    created from `feature_1` and `feature_2`. It checks if any expected frequency is less 
    than 5, indicating a potential issue with using the Chi-square Test. If more than 20% 
    of the expected frequencies are below 5, Fisher's Exact Test is recommended.

    Parameters
    ----------
    feature_1 : pd.Series
        The first categorical feature for comparison.
    feature_2 : pd.Series
        The second categorical feature for comparison.

    Returns
    -------
    bool
        True if Fisher's Exact Test is recommended (more than 20% expected frequencies < 5),
        False if Chi-square Test can be used.
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
    if percent>0.2: return True # requires Fisher's Exact Test
    else: return False          # chi2 test can be used

def chi2_fisher_exact(feature_1:pd.Series, feature_2:pd.Series)->float:

    """
    Perform chi-square test or Fisher's exact test based on expected frequencies.

    This function computes a contingency table (crosstab) from `feature_1` and `feature_2` 
    and determines whether to use Fisher's exact test or chi-square test based on the 
    expected frequencies in the contingency table.

    Parameters
    ----------
    feature_1 : pd.Series
        The first categorical feature for comparison.
    feature_2 : pd.Series
        The second categorical feature for comparison.

    Returns
    -------
    float
        The p-value resulting from either the chi-square test or Fisher's exact test.

    Notes
    -----
    If the expected frequencies in any cell of the contingency table are less than 5, 
    Fisher's exact test is used. Otherwise, the chi-square test is used.
    """

    crosstab = pd.crosstab(feature_1, feature_2)

    if decide_fisher_chi2(feature_1, feature_2):
        return stats.fisher_exact(crosstab).pvalue
    else:
        return stats.chi2_contingency(crosstab, correction=False).pvalue

def chi_squared_tests(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    """
    Perform chi-square or Fisher's exact tests for multiple variables against a grouping variable.

    This function computes p-values for each variable in `variables` against `group_var` using either
    chi-square tests or Fisher's exact tests based on expected frequencies in contingency tables.

    Parameters
    ----------
    df_data : pd.DataFrame
        The pandas DataFrame containing the data.
    variables : list
        A list of column names (variables) in `df_data` to test against `group_var`.
    group_var : str
        The column name in `df_data` representing the grouping variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the variables and their corresponding p-values from the tests.

    Notes
    -----
    This function internally calls `chi2_fisher_exact` for each variable to determine the appropriate
    statistical test based on expected frequencies in contingency tables.
    """

    crosstab_results = {}
    for var in variables:

        pval = chi2_fisher_exact(df_data[var], df_data[group_var])
        crosstab_results[var] = {'p_val': pval}

    result = pd.DataFrame(crosstab_results).transpose()
    result = result.reset_index()

    result.columns = ['Variable', 'p-value']

    return result

def count_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    """
    Calculate counts and percentages of non-null values for each feature in the DataFrame.

    This function computes the total count of non-null values, percentage of non-null values,
    and formats the result for each feature in `features` of the provided DataFrame `data`.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing the data.
    features : list
        A list of column names (features) in `data` for which counts and percentages are computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'Variable' and 'Total', where 'Variable' contains feature names
        and 'Total' contains formatted strings of counts and percentages.

    Notes
    -----
    The percentage is computed as (count of non-null values / total non-null values) * 100.
    Null values (NaNs) are ignored in the count and percentage computation.
    """

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        count = int(np.nansum(data[feat]))
        non_null = (~data[feat].isnull()).sum()
        percent = round(100*(count/non_null),1)

        result.loc[feat, 'Total'] = f"{count} ({percent})"

    result = result.reset_index()

    result.columns = ['Variable', 'Total']

    return result

def summaryze_count_percent(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, groups:list)->pd.DataFrame:

    num_rows = df_sum.shape[0]

    group_1 = groups[0]
    group_2 = groups[1]

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
        df_sum.loc[num_rows+k,'Available Samples for Analysis'] = f"{count[group_1][0]+count[group_2][0]}"

    return df_sum

def median_iqr_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        first = round(np.nanquantile(data[feat], 0.25),1)
        median = round(np.nanquantile(data[feat], 0.5),1)
        third = round(np.nanquantile(data[feat], 0.75),1)

        result.loc[feat, 'Total'] = f"{median} ({first} - {third})"

    result = result.reset_index()

    result.columns = ['Variable', 'Total']

    return result

def mann_whitney(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    mw_results = {}

    groups = df_data[group_var].unique().tolist()

    for var in variables:
        group1 = df_data[df_data[group_var] == groups[0]][var]
        group2 = df_data[df_data[group_var] == groups[1]][var]
        u_stat, p_val = stats.mannwhitneyu(group1.dropna(), group2.dropna())
        mw_results[var] = {'u_stat': u_stat, 'p_val': p_val}

    results = pd.DataFrame(mw_results).transpose().reset_index().drop(columns='u_stat')

    results.columns = ['Variable', 'p-value']

    results['p-value'] = results['p-value'].apply(
        lambda x: str(round(x,4)) if x > 0.001 else "p<0.001"
    )

    return results

def median_iqr(data:pd.DataFrame, features:list, grouping_by:str)->pd.DataFrame:

    def first_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.25)
    def median(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.5)
    def third_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.75)

    agg_dict = {feat: [first_Q, median, third_Q, 'count'] for feat in features}

    grouped = data.groupby(by=grouping_by, as_index=False)[features]\
        .agg(agg_dict)\
        .transpose()\
        .reset_index(drop=False)
    
    grouped.columns = grouped.loc[0,:].to_list()
    grouped.columns = ['Variable', 'Stat'] + list(grouped.columns[2:])
    grouped = grouped.loc[1:,:].copy()
    
    return grouped

def summaryze_median_iqr(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, group_1:str, group_2:str)->pd.DataFrame:

    num_rows = df_sum.shape[0]

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
        df_sum.loc[num_rows+k,'Available Sample for Analysis'] = f"{count[group_1][0]+count[group_2][0]}"

    return df_sum

def count_categories(data:pd.DataFrame, grouping_by:str, variable:str, new_var_name:str=None)->pd.DataFrame:

    groups = data[grouping_by].unique().tolist()

    df_0 = data[data[grouping_by]==groups[0]].reset_index(drop=True)
    df_1 = data[data[grouping_by]==groups[1]].reset_index(drop=True)

    count_0 = df_0[variable].value_counts().reset_index()
    count_0.columns = [variable, groups[0]]

    count_1 = df_1[variable].value_counts().reset_index()
    count_1.columns = [variable, groups[1]]

    result = pd.merge(count_0, count_1, on=variable)

    result['Total'] = result[groups[0]] + result[groups[1]]

    result[groups[0]] = result[groups[0]].apply(
        lambda x: f"{x} ({round(100*x/df_0.shape[0], 1)})"
    )

    result[groups[1]] = result[groups[1]].apply(
        lambda x: f"{x} ({round(100*x/df_1.shape[0], 1)})"
    )

    result['Total'] = result['Total'].apply(
        lambda x: f"{x} ({round(100*x/df_1.shape[0], 1)})"
    )

    if new_var_name is not None:
        result.columns = [col if col!=variable else new_var_name for col in result.columns]

    return result

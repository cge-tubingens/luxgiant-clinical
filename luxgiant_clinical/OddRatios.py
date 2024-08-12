"""
Module with functions tailored to get unadjusted odd ratios and adjusted odd ratios for matched data.
"""

import pandas as pd
import numpy as np

def mcnemar_table(data:pd.DataFrame, df_matched:pd.DataFrame, feature_col:str, id_col:str, marginals:bool=False)->pd.DataFrame:

    """
    Constructs a contingency table for McNemar's test from matched case-control data.

    This function merges patient and control data, based on the specified feature column, to create 
    a contingency table used for McNemar's test. The table shows the number of discordant pairs 
    where the case (patient) and control have differing feature values, which is useful in evaluating 
    the difference in proportions for paired binary data.

    Parameters:
    -----------
    data: pd.DataFrame 
        The original dataset containing feature information.
    df_matched: pd.DataFrame 
        A DataFrame containing matched case-control pairs with columns 'Patient_id' and 'Control_id'.
    feature_col: str 
        The column name in `data` representing the binary feature of interest.
    id_col: str 
        The column name in `data` that identifies unique individuals (e.g., patient ID).
    marginals: bool (optional) 
        If True, adds marginal sums (totals) to the contingency table. Defaults to False.

    Returns:
    --------
    pd.DataFrame: A 2x2 contingency table with the counts of case-control pairs for the following:
                - Case_1, Control_1: Both case and control have the feature.
                - Case_1, Control_0: Case has the feature, control does not.
                - Case_0, Control_1: Control has the feature, case does not.
                - Case_0, Control_0: Neither case nor control have the feature.
                If `marginals` is True, a 'Total' row and column are added with the sums of each row and column.
    """

    # collect information from patients
    df_temp = pd.merge(df_matched, data, left_on='Patient_id', right_on=id_col)
    df_temp  = df_temp.rename(columns={feature_col: f"{feature_col}_ptnt"})

    # collect information from controls
    df_temp = pd.merge(df_temp, data, left_on='Control_id', right_on=id_col)
    df_temp  = df_temp.rename(columns={feature_col: f"{feature_col}_ctrl"})

    # drop redundant columns
    cols_to_drop = [col for col in df_temp.columns if '_id' in col]

    df_temp = df_temp\
        .drop(columns=cols_to_drop)\
        .dropna(inplace=False, ignore_index=True, how='any')
    
    # determine patient and control columns
    for col in df_temp.columns:
        if '_ptnt' in col: ptnt_col = col
        if '_ctrl' in col: ctrl_col = col

    # create mask for positive risk factors
    mask_ptnt_pos = (df_temp[ptnt_col]==1)
    mask_ctrl_pos = (df_temp[ctrl_col]==1)

    # create empty dataframe
    df = pd.DataFrame(columns=['Control_1', 'Control_0'], index=['Case_1', 'Case_0']) 
    
    # fill values of the "contingency table"
    df.loc['Case_1', 'Control_1'] = (mask_ptnt_pos & mask_ctrl_pos).sum()
    df.loc['Case_0', 'Control_0'] = (~mask_ptnt_pos & ~mask_ctrl_pos).sum()
    df.loc['Case_1', 'Control_0'] = (mask_ptnt_pos & ~mask_ctrl_pos).sum()
    df.loc['Case_0', 'Control_1'] = (~mask_ptnt_pos & mask_ctrl_pos).sum()

    # add marginals value to "contingency table" if wanted
    if marginals:

        df.loc['Total', 'Control_1'] = mask_ctrl_pos.sum()
        df.loc['Total', 'Control_0'] = (~mask_ctrl_pos).sum()

        df['Total'] = df['Control_1']+df['Control_0']

    return df

def mcnemar_test(paired_cont:pd.DataFrame)-> tuple:

    """
    Performs McNemar's test on a paired 2x2 contingency table and calculates the odds ratio with its 
    confidence interval.

    This function computes McNemar's test statistic and p-value from a given 2x2 contingency table that represents
    paired data (e.g., before and after treatment). Additionally, it calculates the odds ratio and its 95% 
    confidence interval, which provide insights into the effect size and its precision.

    Parameters:
    -----------
    paired_cont: 
        pd.DataFrame A 2x2 contingency table as a DataFrame, where:
            - Row 0, Column 1 represents `b` (case has feature, control does not).
            - Row 1, Column 0 represents `c` (control has feature, case does not).

    Returns:
    --------
    tuple: 
        A tuple containing the following elements:
            - float: Odds ratio, rounded to two decimal places.
            - tuple: 95% confidence interval for the odds ratio, rounded to two decimal places.
            - float: McNemar's test statistic.
            - float: p-value for McNemar's test.
    """

    from math import log, sqrt, exp
    
    from statsmodels.stats.contingency_tables import mcnemar

    # determine elements off the main diagonal
    b = paired_cont.iloc[0,1]
    c = paired_cont.iloc[1,0]

    # add small correction if some of the values is 0
    if b==0 or c==0:
        b+=0.5
        c+=0.5

    # compute McNemar test
    mcnemar_res = mcnemar(paired_cont.to_numpy(), exact=True, correction=False)

    statistic = mcnemar_res.statistic
    p = mcnemar_res.pvalue

    # compute odds ratio and its logarithm
    odds_ratio = b/c
    log_odds = log(odds_ratio)

    # compute confidence interval
    se_odds = sqrt(1/b+1/c)
    log_conf_int= (log_odds - 1.96*se_odds, log_odds + 1.96*se_odds)

    conf_int = (np.round(exp(log_conf_int[0]),2), np.round(exp(log_conf_int[1]), 2))

    return np.round(odds_ratio, 2), conf_int, statistic, p

def report_mcnemar(data:pd.DataFrame, df_matched:pd.DataFrame, variables:list, id_col:str)->pd.DataFrame:

    """
    Generates a summary report of McNemar's test results for a set of variables.

    This function computes McNemar's test for multiple binary variables from matched case-control data and compiles
    the results into a summary DataFrame. For each variable, the function calculates the odds ratio, its 95% confidence
    interval, and the p-value, which are then formatted and reported.

    Parameters:
    -----------
    data: pd.DataFrame 
        The original dataset containing feature information, with one row per individual.
    df_matched: pd.DataFrame 
        A DataFrame containing matched case-control pairs, typically with columns 'Patient_id' and 'Control_id'.
    variables: list 
        A list of binary feature column names in `data` for which McNemar's test will be performed.
    id_col: str 
        The column name in `data` that identifies unique individuals (e.g., patient ID).

    Returns:
    --------
    pd.DataFrame: 
        A DataFrame with the following columns:
        - "McN OR (95%CI)": The odds ratio and its 95% confidence interval for each variable.
        - "p-value": The p-value from McNemar's test, with "<0.001" notation if the p-value is extremely small.
        The index of the DataFrame will be the variable names.
    """

    # create empty dataframe
    report = pd.DataFrame(columns=["McN OR (95%CI)", "p-value"], index=variables)

    # loop to fill the values of the dataframe
    for var in variables:

        mcn_table = mcnemar_table(
            data       =data[[id_col, var]], 
            df_matched =df_matched, 
            feature_col=var, 
            id_col     =id_col
        )

        OR, CI, stat, pval = mcnemar_test(mcn_table)

        report.loc[var, "McN OR (95%CI)"] = f"{OR} {CI}"

        # format results
        if pval<0.001:
            report.loc[var, "p-value"] = "p<0.001"
        else:
            report.loc[var, "p-value"] = f"{np.round(pval,3)}"

    report = report.reset_index()
    report.columns = ['Variables', 'McN OR (95% CI)', 'p-value']

    return report

def adjusted_odds_ratios(data:pd.DataFrame, target:str, target_code:dict, variables:list, match_1:str, match_2:str)->pd.DataFrame:

    from statsmodels.discrete.conditional_models import ConditionalLogit

    def pval_format(pval:float)->str:

        if pval<0.001: return 'p<0.001'
        else: return np.round(pval, 3)

    X_copy = data.copy()

    # recode target variable
    X_copy[target] = X_copy[target].map(target_code)

    # create matched data identifier
    X_copy['Group ID'] = X_copy[match_1].astype(str) + '_' + X_copy[match_2].astype(str)

    # drop any sample with missing values
    X = X_copy.dropna(subset=variables, how='any', ignore_index=True, inplace=False)

    #for var in variables:
    #    X[var] = X[var].apply(lambda x: int(x))

    # train conditional logistic regression model
    clogit = ConditionalLogit(endog=X[target], exog=X[variables], groups=X['Group ID'])
    
    results = clogit.fit()

    odds = pd.DataFrame(columns=['Adjusted OR', 'Lower', 'Upper', 'p-value'], index=variables)

    odds['Adjusted OR'] = np.round(np.exp(results.params.values), 2)
    odds['Lower']       = np.round(np.exp(results.conf_int()[0].values), 2)
    odds['Upper']       = np.round(np.exp(results.conf_int()[1].values), 2)
    odds['p-value']     = results.pvalues.values

    odds['Adjusted OR (95%) CI'] = odds['Adjusted OR'].astype(str) + ' (' \
        + odds['Lower'].astype(str) + ', ' + odds['Upper'].astype(str) + ')'
    
    odds = odds.drop(columns=['Adjusted OR', 'Lower', 'Upper'], inplace=False)
    odds['p-value'] = odds['p-value'].apply(lambda x: pval_format(x))

    odds = odds.reset_index()
    odds.columns = ['Variables', 'Adjusted OR (95% CI)', 'p-value']

    return odds

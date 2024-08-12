"""
Module with functions tailored to get unadjusted odd ratios and adjusted odd ratios for matched data.
"""

import pandas as pd
import numpy as np

def mcnemar_table(data:pd.DataFrame, df_matched:pd.DataFrame, feature_col:str, id_col:str, marginals:bool=False)->pd.DataFrame:

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

    report = pd.DataFrame(columns=["McN OR (95%CI)", "p-value"], index=variables)

    for var in variables:

        mcn_table = mcnemar_table(
            data=data[[id_col, var]], 
            df_matched=df_matched, 
            feature_col=var, 
            id_col=id_col
        )

        OR, CI, stat, pval = mcnemar_test(mcn_table)

        report.loc[var, "McN OR (95%CI)"] = f"{OR} {CI}"
        if pval<0.001:
            report.loc[var, "p-value"] = "p<0.001"
        else:
            report.loc[var, "p-value"] = f"{np.round(pval,3)}"

    return report

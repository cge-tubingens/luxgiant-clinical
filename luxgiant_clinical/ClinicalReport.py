
import os
import pandas as pd
import numpy as np

from scipy import stats

def contingency_table(X:pd.DataFrame, index_col:str, columns:str, outputPath:str)->pd.DataFrame:

    contingency = pd.crosstab(X[index_col], X[columns])\
        .reset_index()\
        .rename_axis(None, axis=1)

    contingency.to_csv(outputPath)

    return contingency

def descriptive_analytics(df_data:pd.DataFrame, variables:list)->pd.DataFrame:

    frequency_stats = {}
    for var in variables:
        desc = df_data[var].describe()
        frequency_stats[var] = {
            'mean'  : np.round(desc['mean'], 3),
            'stddev': np.round(desc['std'],3),
            'min'   : np.round(desc['min'],3),
            'max'   : np.round(desc['max'],3),
            'first_Q': np.round(desc['25%'],3),
            'median': np.round(desc['50%'],3),
            'third_Q': np.round(desc['75%'],3)
        }

    return pd.DataFrame(frequency_stats)

def descriptive_analytics_by_group(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    def first_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.25)
    def median(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.5)
    def third_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.75)
    
    all_vars = variables + [group_var]

    df_grouped = df_data[all_vars].groupby(by=group_var).agg(
       ['mean', 'std', 'min', 'max', first_Q, median, third_Q]
    )

    return df_grouped.transpose()

def conditioned_descriptive_analytics_by_group(df_data:pd.DataFrame, variables:list, group_var:str, condition:tuple)->pd.DataFrame:
    
    def first_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.25)
    def median(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.5)
    def third_Q(x:pd.DataFrame)->float: 
        return np.nanquantile(x, 0.75)
    
    df_filtered = df_data[df_data[condition[0]]==condition[1]].reset_index(drop=True)
    
    all_vars = variables + [group_var]

    df_grouped = df_filtered[all_vars].groupby(by=group_var).agg(
       ['mean', 'std', 'min', 'max', first_Q, median, third_Q]
    )

    return df_grouped.transpose()

def t_test_by_group(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    df_data_new = df_data[~df_data[group_var].isnull()].reset_index(drop=True)

    ttest_results = {}
    groups = df_data_new[group_var].unique().tolist()
    
    for var in variables:
        group1 = df_data_new[df_data_new[group_var] == groups[0]][var]
        group2 = df_data_new[df_data_new[group_var] == groups[1]][var]
        t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())
        ttest_results[var] = {'t_stat': t_stat, 'p_val': p_val}

    return pd.DataFrame(ttest_results)

def mann_whitney_by_groups(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    mw_results = {}

    df_data_new = df_data[~df_data[group_var].isnull()].reset_index(drop=True)
    groups = df_data_new[group_var].unique().tolist()

    for var in variables:
        group1 = df_data_new[df_data_new[group_var] == groups[0]][var]
        group2 = df_data_new[df_data_new[group_var] == groups[1]][var]
        u_stat, p_val = stats.mannwhitneyu(group1.dropna(), group2.dropna())
        mw_results[var] = {'u_stat': u_stat, 'p_val': p_val}

    return pd.DataFrame(mw_results)

def conditioned_mann_whitney_by_groups(df_data:pd.DataFrame, variables:list, group_var:str, condition:tuple)->pd.DataFrame:

    mw_results = {}
    df_data_new = df_data[~df_data[group_var].isnull()].reset_index(drop=True)
    groups = df_data_new[group_var].unique().tolist()

    df_filtered = df_data_new[df_data_new[condition[0]]==condition[1]].reset_index(drop=True)

    for var in variables:
        group1 = df_filtered[df_filtered[group_var] == groups[0]][var]
        group2 = df_filtered[df_filtered[group_var] == groups[1]][var]
        u_stat, p_val = stats.mannwhitneyu(group1.dropna(), group2.dropna())
        mw_results[var] = {'u_stat': u_stat, 'p_val': p_val}

    return pd.DataFrame(mw_results)

def chi_squared_tests(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    crosstab_results = {}
    for var in variables:
        crosstab = pd.crosstab(df_data[var], df_data[group_var])
        chi2, p, dof, ex = stats.chi2_contingency(crosstab)
        crosstab_results[var] = {'chi2': chi2, 'p_val': p, 'dof': dof}


    return pd.DataFrame(crosstab_results)

def chi_squared_conditioned(df_data:pd.DataFrame, conditionants:dict, condition:str, group_var:str)->pd.DataFrame:

    df_res = pd.DataFrame(columns=['Condition', 'Variable', 'chi2', 'p_val', 'dof'])

    count = 0

    for key in conditionants.keys():

        var_list = conditionants[key]

        for var in var_list:

            mask = (df_data[key]==condition)

            df_cond = df_data[mask].reset_index(drop=True)

            temp = chi_squared_tests(df_data=df_cond, variables=[var], group_var=group_var)

            df_res.loc[count, 'Condition']= key
            df_res.loc[count, 'Variable'] = var
            df_res.loc[count, 'chi2']     = temp.loc['chi2', var]
            df_res.loc[count, 'p_val']    = temp.loc['p_val', var]
            df_res.loc[count, 'dof']      = temp.loc['dof', var]

            count+=1

    return df_res

def fisher_exact_test(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    crosstab_results = {}
    for var in variables:
        crosstab = pd.crosstab(df_data[var], df_data[group_var])
        fisher, p= stats.fisher_exact(crosstab)
        crosstab_results[var] = {'fisher': fisher, 'p_val': p}

    return pd.DataFrame(crosstab_results)

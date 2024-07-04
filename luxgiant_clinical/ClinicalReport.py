
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

from scipy import stats

from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

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

    def fisher_exact_R():

        return None

    crosstab_results = {}
    for var in variables:
        crosstab = pd.crosstab(df_data[var], df_data[group_var])
        fisher, p= stats.fisher_exact(crosstab)
        crosstab_results[var] = {'fisher': fisher, 'p_val': p}

    return pd.DataFrame(crosstab_results)

def oneway_anova(df_data:pd.DataFrame, variables:list, group_var:str, condition:tuple=None)->pd.DataFrame:

    if condition is not None:
        df_filtered = df_data[df_data[condition[0]]==condition[1]].reset_index(drop=True)
    else:
        df_filtered = df_data.copy()

    anova_results = {
        var: sm.stats.anova_lm(ols(f'{var} ~ C({group_var})', data=df_filtered).fit(), typ=2) for var in variables
    }

    multi = pd.MultiIndex.from_product(
        iterables=[variables, [f'C({group_var})', 'Residual']], 
        names    =['variable', 'paramters']
    )
    cols = ['sum_sq', 'df', 'F', 'PR(>F)']

    df_anova = pd.DataFrame(index=multi, columns=cols)

    for col in cols:
        for index in multi:
            df_anova.loc[index, col]=anova_results[index[0]].loc[index[1], col]

    return df_anova

def crossed_elements(groups:list)->list:

    cross = []

    while len(groups)>0:
        cart = [(groups[0],y) for y in groups[1:]]
        for elem in cart:
            cross.append(elem)
        groups = groups[1:]

    return cross

def bonferroni_correction_one_variable(df_data:pd.DataFrame, variable:str, group_var:str)->pd.DataFrame:

    classes = df_data[group_var].unique()

    df_clean = df_data[~df_data[variable].isnull()].reset_index(drop=True)
    df_clean = df_clean[[variable, group_var]].copy()

    n_samples = df_clean.shape[0]

    t_tests = pd.DataFrame(columns=['Group', 't', 'p_value', 'n_samples', 'adj_p'])

    cartesian = crossed_elements(classes)

    for k,elem in enumerate(cartesian):

        group_1 = df_clean[df_clean[group_var]==elem[0]].reset_index(drop=True)[variable]
        group_2 = df_clean[df_clean[group_var]==elem[1]].reset_index(drop=True)[variable]
        t, p = ss.ttest_ind(group_1, group_2)

        t_tests.loc[k, 'Group']= f"{elem[0]} vs {elem[1]}"
        t_tests.loc[k, 't']        = t
        t_tests.loc[k, 'p_value']  = p
        t_tests.loc[k, 'n_samples']= n_samples
        t_tests.loc[k, 'adj_p']    = p*len(classes)

    return t_tests

def multi_variable_bonferroni(df_data:pd.DataFrame, variables:str, group_var:str, condition:tuple=None)->pd.DataFrame:

    if condition is not None:
        df_filtered = df_data[df_data[condition[0]]==condition[1]].reset_index(drop=True)
    else:
        df_filtered = df_data.copy()
    
    results = pd.DataFrame()

    for var in variables:

        temp = bonferroni_correction_one_variable(
            df_data  =df_filtered,
            variable =var,
            group_var=group_var
        )

        temp['Variable'] = var

        results = pd.concat([results, temp], axis=0, ignore_index=True)

    return results.set_index(['Variable', 'Group'])

def multi_variable_kw(df_data:pd.DataFrame, variables:str, group_var:str, condition:tuple=None)->pd.DataFrame:

    if condition is not None:
        df_filtered = df_data[df_data[condition[0]]==condition[1]].reset_index(drop=True)
    else:
        df_filtered = df_data.copy()

    classes = df_filtered[group_var].unique()
    kw_tests = pd.DataFrame(columns=['Variable', 'kw', 'p_value', 'n_samples'])

    for k, var in enumerate(variables):

        df_clean = df_filtered[~df_filtered[var].isnull()].reset_index(drop=True)
        df_clean = df_clean[[var, group_var]].copy()

        n_samples = df_clean.shape[0]

        groups = [
            df_clean[df_clean[group_var]==elem][var].reset_index(drop=True) for elem in classes
        ]

        kw, p = ss.kruskal(*groups)

        kw_tests.loc[k, 'Variable'] = var
        kw_tests.loc[k, 'kw']       = kw
        kw_tests.loc[k, 'p_value']  = p
        kw_tests.loc[k, 'n_samples']= n_samples

    kw_tests.name = f"Kruskal-Wallis test for groups of variable {group_var} "

    return kw_tests

def intergroup_kw_with_bonferroni_correction_one_variable(df_data:pd.DataFrame, variable:str, group_var:str)->pd.DataFrame:

    classes = df_data[group_var].unique()

    df_clean = df_data[~df_data[variable].isnull()].reset_index(drop=True)
    df_clean = df_clean[[variable, group_var]].copy()

    n_samples = df_clean.shape[0]

    kw_tests = pd.DataFrame(columns=['Group', 'kw', 'p_value', 'n_samples', 'adj_p'])

    cartesian = crossed_elements(classes)

    for k,elem in enumerate(cartesian):

        group_1 = df_clean[df_clean[group_var]==elem[0]].reset_index(drop=True)[variable]
        group_2 = df_clean[df_clean[group_var]==elem[1]].reset_index(drop=True)[variable]
        t, p = ss.kruskal(group_1, group_2)

        kw_tests.loc[k, 'Group']= f"{elem[0]} vs {elem[1]}"
        kw_tests.loc[k, 'kw']        = t
        kw_tests.loc[k, 'p_value']  = p
        kw_tests.loc[k, 'n_samples']= n_samples
        kw_tests.loc[k, 'adj_p']    = p*len(classes)

    return kw_tests

def multi_variable_kw_bonferroni(df_data:pd.DataFrame, variables:str, group_var:str, condition:tuple=None)->pd.DataFrame:

    if condition is not None:
        df_filtered = df_data[df_data[condition[0]]==condition[1]].reset_index(drop=True)
    else:
        df_filtered = df_data.copy()
    
    results = pd.DataFrame()

    for var in variables:

        temp = intergroup_kw_with_bonferroni_correction_one_variable(
            df_data  =df_filtered,
            variable =var,
            group_var=group_var
        )

        temp['Variable'] = var

        results = pd.concat([results, temp], axis=0, ignore_index=True)

    return results.set_index(['Variable', 'Group'])

def chi_squared_analysis(df_data:pd.DataFrame, conditionants:list, variables_4_analysis:list, grouping_var:str)->pd.DataFrame:

    # create copy of data frame
    df_copy = df_data.copy()

    if conditionants is not None:

        # list for conditioned features
        conditioned_feat = []

        for k, condition in enumerate(conditionants):

            # filter data frame by conditions
            df_copy = df_copy[df_copy[condition[0]]==condition[1]].reset_index(drop=True)

            conditioned_feat.append(f"Conditioned Feat {k+1}")
            conditioned_feat.append(f"Condition {k+1}")

        # create empy data frame
        df_res = pd.DataFrame(columns=conditioned_feat + ['Variable', 'chi2', 'p_val', 'dof'])

    else:
        df_res = pd.DataFrame(columns=['Variable', 'chi2', 'p_val', 'dof'])


    for m, var in enumerate(variables_4_analysis):

        # write column values for filter variable and condition
        if conditionants is not None:
            for k, condition in enumerate(conditionants):
                df_res.loc[m, f"Conditioned Feat {k+1}"]= condition[0]
                df_res.loc[m, f"Condition {k+1}"]= condition[1]

        # perform a chi squared test
        temp = chi_squared_tests(df_data=df_copy, variables=[var], group_var=grouping_var)

        # write remaing column values
        df_res.loc[m, 'Variable'] = var
        df_res.loc[m, 'chi2']     = temp.loc['chi2', var]
        df_res.loc[m, 'p_val']    = temp.loc['p_val', var]
        df_res.loc[m, 'dof']      = temp.loc['dof', var]

    return df_res

def exact_fisher_analysis(df_data:pd.DataFrame, conditionants:list, variables_4_analysis:list, grouping_var:str)->pd.DataFrame:

    # create copy of data frame
    df_copy = df_data.copy()

    if conditionants is not None:

        # list for conditioned features
        conditioned_feat = []

        for k, condition in enumerate(conditionants):

            # filter data frame by conditions
            df_copy = df_copy[df_copy[condition[0]]==condition[1]].reset_index(drop=True)

            conditioned_feat.append(f"Conditioned Feat {k+1}")
            conditioned_feat.append(f"Condition {k+1}")

        # create empy data frame
        df_res = pd.DataFrame(columns=conditioned_feat + ['Variable', 'chi2', 'p_val', 'dof'])

    else:
        # create empy data frame
        df_res = pd.DataFrame(columns=['Variable', 'chi2', 'p_val', 'dof'])


    for m, var in enumerate(variables_4_analysis):

        # write column values for filter variable and condition
        if conditionants is not None:
            for k, condition in enumerate(conditionants):
                df_res.loc[m, f"Conditioned Feat {k+1}"]= condition[0]
                df_res.loc[m, f"Condition {k+1}"]= condition[1]

        # perform a Fisher exact test
        temp = fisher_exact_test(df_data=df_copy, variables=[var], group_var=grouping_var)
        print(var)

        # write remaing column values
        df_res.loc[m, 'Variable'] = var
        df_res.loc[m, 'fisher']   = temp.loc['fisher', var]
        df_res.loc[m, 'p_val']    = temp.loc['p_val', var]

    return df_res

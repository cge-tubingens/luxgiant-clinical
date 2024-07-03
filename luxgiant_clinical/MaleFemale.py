"""
Module with function tailored to 
"""

import numpy as np
import pandas as pd

from scipy import stats

def t_test_by_group(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    ttest_results = {}
    groups = df_data[group_var].unique().tolist()
    
    for var in variables:
        group1 = df_data[df_data[group_var] == groups[0]][var]
        group2 = df_data[df_data[group_var] == groups[1]][var]
        t_stat, p_val = stats.ttest_ind(group1.dropna(), group2.dropna())
        ttest_results[var] = {'t_stat': t_stat, 'p_value': p_val}

    results = pd.DataFrame(ttest_results).transpose().drop(columns='t_stat')
    results['p_value'] = results['p_value'].apply(
        lambda x: str(round(x,4)) if x > 0.001 else "p<0.001"
    )

    results = results.reset_index()
    results.columns = ['Variable', 'p-value']

    return results

def mean_std_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        mean= round(np.mean(data[feat]),2)
        std = round(np.std(data[feat]), 2)

        result.loc[feat, 'Total'] = f"{mean} ({std})"

    result = result.reset_index()

    result.columns = ['Variable', 'Total']

    return result

def mean_std(data:pd.DataFrame, features:list, grouping_by:str)->pd.DataFrame:

    agg_dict = {feat: ['mean', 'std', 'count'] for feat in features}

    grouped = data.groupby(by=grouping_by, as_index=False)[features]\
        .agg(agg_dict)\
        .transpose()\
        .reset_index(drop=False)
    
    grouped.columns = grouped.loc[0,:].to_list()
    grouped.columns = ['Variable', 'Stat'] + list(grouped.columns[2:])
    grouped = grouped.loc[1:,:].copy()

    return grouped

def summaryze_mean_std(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, group_1:str, group_2:str)->pd.DataFrame:

    num_rows = df_sum.shape[0]

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
        df_sum.loc[num_rows+k,group_1] = f"{round(mean[group_1][0],2)} ({round(std[group_1][0],2)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(mean[group_2][0],2)} ({round(std[group_2][0],2)})"
        df_sum.loc[num_rows+k,'Available Sample for Analysis'] = f"{count[group_1][0]+count[group_2][0]}"

    return df_sum

def count_percent(data:pd.DataFrame, features:list, grouping_by:str)->pd.DataFrame:

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

def chi_squared_tests(df_data:pd.DataFrame, variables:list, group_var:str)->pd.DataFrame:

    crosstab_results = {}
    for var in variables:
        crosstab = pd.crosstab(df_data[var], df_data[group_var])
        try:
            chi2, p, dof, ex = stats.chi2_contingency(crosstab)
        except ValueError:
            crosstab_results[var] = {'chi2': np.nan, 'p_val': np.nan, 'dof': np.nan}
        else:
            crosstab_results[var] = {'chi2': chi2, 'p_val': p, 'dof': dof}

    result = pd.DataFrame(crosstab_results).transpose().drop(columns=['chi2', 'dof'], inplace=False)
    result = result.reset_index()

    result.columns = ['Variable', 'p-value']

    result['p-value'] = result['p-value'].apply(
        lambda x: str(round(x,4)) if x > 0.001 else "p<0.001"
    )

    return result

def count_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        count = int(np.nansum(data[feat]))
        percent = round(100*(count/data.shape[0]),2)

        result.loc[feat, 'Total'] = f"{count} ({percent})"

    result = result.reset_index()

    result.columns = ['Variable', 'Total']

    return result

def summaryze_count_percent(df_sum:pd.DataFrame, df_grouped:pd.DataFrame, variables:list, group_1:str, group_2:str)->pd.DataFrame:

    num_rows = df_sum.shape[0]

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
        df_sum.loc[num_rows+k,group_1] = f"{round(sum[group_1][0])} ({round(perc[group_1][0],2)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(sum[group_2][0])} ({round(perc[group_2][0],2)})"
        df_sum.loc[num_rows+k,'Available Sample for Analysis'] = f"{count[group_1][0]+count[group_2][0]}"

    return df_sum

def median_iqr_simple(data:pd.DataFrame, features:list)->pd.DataFrame:

    result = pd.DataFrame(index=features, columns=['Total'])

    for feat in features:

        first = round(np.nanquantile(data[feat], 0.25),2)
        median = round(np.nanquantile(data[feat], 0.5),2)
        third = round(np.nanquantile(data[feat], 0.75),2)

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
        df_sum.loc[num_rows+k,group_1] = f"{round(medn[group_1][0],2)} ({round(fstq[group_1][0],2)} - {round(trdq[group_1][0],2)})"
        df_sum.loc[num_rows+k,group_2] = f"{round(medn[group_2][0],2)} ({round(fstq[group_2][0],2)} - {round(trdq[group_2][0],2)})"
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
        lambda x: f"{x} ({round(100*x/df_0.shape[0], 2)})"
    )

    result[groups[1]] = result[groups[1]].apply(
        lambda x: f"{x} ({round(100*x/df_1.shape[0], 2)})"
    )

    result['Total'] = result['Total'].apply(
        lambda x: f"{x} ({round(100*x/df_1.shape[0], 2)})"
    )

    if new_var_name is not None:
        result.columns = [col if col!=variable else new_var_name for col in result.columns]

    return result

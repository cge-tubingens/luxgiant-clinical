
import argparse

import pandas as pd

def categories_recoder(data_df:pd.DataFrame, features:list, code:dict)->pd.DataFrame:

    """
    Recode categorical variables in a DataFrame based on a provided mapping.

    This function applies a specified recoding to a list of categorical features in a DataFrame. 
    The recoding is done using a dictionary that maps the original categories to new values. The 
    DataFrame is modified in place and returned with the updated values.

    Parameters:
    -----------
    data_df: pd.DataFrame 
        The DataFrame containing the data to be recoded.
    features: list 
        A list of column names in `data_df` representing the categorical features to be recoded.
    code: dict 
        A dictionary where keys are the original category values and values are the new category codes.

    Returns:
    --------
    pd.DataFrame: 
        The DataFrame with the specified features recoded based on the provided mapping.
    """

    for feat in features:
        data_df[feat] = data_df[feat].map(code)

    return data_df


def recover_columns_names(columns:list)->list:

    """
    Recover the original column names from a list of transformed column names.

    Parameters:
    -----------
    columns : list
        List of transformed column names.

    Returns:
    --------
    old_columns : list
        List of original column names.
    """

    old_columns = []

    for col in columns:
        splitted = col.split('__')
        if len(splitted)==2:
            old_columns.append(splitted[1])
        else:
            old_columns.append(splitted[1] + '__' + splitted[2])
    return old_columns

def arg_parser()->dict:

    # define parser
    parser = argparse.ArgumentParser(description='Adresses to input STATA file and output folder')

    # parameters of quality control
    parser.add_argument('--input-file', type=str, nargs='?', default=None, const=None, help='Full path to the STATA file with REDCap raw data.')

    # path to data and names of files
    parser.add_argument('--output-folder', type=str, nargs='?', default=None, const=None, help='Full path to the to folder where cleaned file will be saved.')

    # parse args and turn into dict
    args = parser.parse_args()

    return args

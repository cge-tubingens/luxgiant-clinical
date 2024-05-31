
import argparse


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

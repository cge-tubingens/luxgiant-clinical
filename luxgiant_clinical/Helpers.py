



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

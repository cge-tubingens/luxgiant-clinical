
import os
import pandas as pd


def contingency_table(X:pd.DataFrame, index_col:str, columns:str, outputPath:str)->pd.DataFrame:

    contingency = pd.crosstab(X[index_col], X[columns])\
        .reset_index()\
        .rename_axis(None, axis=1)

    contingency.to_csv(outputPath)

    return contingency

from sklearn.base import TransformerMixin, BaseEstimator

import pandas as pd
import numpy as np

class AgeCategory(BaseEstimator, TransformerMixin):

    def __init__(self, age_cutoff:int, output_col:str) -> None:
        super().__init__()
        self.age_cutoff = age_cutoff
        self.output_col = output_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy    = X.copy()
        col       = X_copy.columns
        age_cutoff= self.age_cutoff
        output_col= self.output_col

        X_copy[output_col] = X_copy[col[0]].apply(
            lambda x: self.encoder(x, age_cutoff)
        )

        return X_copy[[output_col]]
    
    @staticmethod
    def encoder(age:int, cutoff:int)->str:

        if np.isnan(age):
            return None
        
        if age < cutoff:
            return f"<{cutoff} years"
        else:
            return f">={cutoff} years"

class InitialMotorSymptoms(TransformerMixin, BaseEstimator):

    def __init__(self, output_col:str) -> None:
        super().__init__()
        self.output_col = output_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        cols = X_copy.columns
        output_col = self.output_col

        X_copy[output_col] = None

        X_copy.loc[(X_copy[cols[0]] == 'Checked') & (X_copy[cols[1]] == 'Checked'), output_col] = 'Checked'
        X_copy.loc[(X_copy[cols[0]] == 'Unchecked') | (X_copy[cols[1]] == 'Unchecked'), output_col] = 'Unchecked' 

        return X_copy

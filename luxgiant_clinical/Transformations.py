from sklearn.base import TransformerMixin, BaseEstimator

import pandas as pd
import numpy as np

class Identity(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()

        return X_copy

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

class HandYOnOff(TransformerMixin, BaseEstimator):

    def __init__(self, output_col:str='hyonoff') -> None:
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

        X_copy['col_0'] = X_copy[cols[0]].apply(lambda x: x.split(':')[0] if x is not None else None)
        X_copy['col_1'] = X_copy[cols[1]].apply(lambda x: x.split(':')[0] if x is not None else None)

        X_copy.loc[X_copy['col_0'] == 'ON', output_col] = 'On'
        X_copy.loc[X_copy['col_0'] == 'OFF', output_col] = 'Off'

        X_copy.loc[X_copy['col_1'] == 'ON', output_col] = 'On'
        X_copy.loc[X_copy['col_1'] == 'OFF', output_col] = 'Off'

        return X_copy[cols.tolist() + [output_col]]

class PDduration(BaseEstimator, TransformerMixin):

    def __init__(self, output_col:str='pdsl', cutoff:int=5)->None:
        super().__init__()
        self.output_col = output_col
        self.cutoff     = cutoff

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col    = X_copy.columns[0]

        output_col= self.output_col
        cutoff    = self.cutoff

        X_copy[output_col] = None

        X_copy.loc[X_copy[col] <=cutoff, output_col] = f"<={cutoff}"
        X_copy.loc[X_copy[col] >cutoff, output_col] = f">{cutoff}"

        return X_copy

class HandYstage(BaseEstimator, TransformerMixin):

    def __init__(self, output_col:str='hystage')->None:
        super().__init__()
        self.output_col = output_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col    = X_copy.columns[0]

        output_col= self.output_col

        X_copy[output_col] = X_copy[col].apply(lambda x: self.stage_encoder(x))

        return X_copy
    
    @staticmethod
    def stage_encoder(x:str)->str:

        if x is None: return None

        x_num = float(x.split(' - ')[0])

        if x_num <= 3: return 'Not severe'
        else: return 'Severe'

class ExposurePesticide(TransformerMixin, BaseEstimator):

    def __init__(self, output_col) -> None:
        super().__init__()
        self.output_col = output_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()
        cols  = X_copy.columns

        output_col = self.output_col

        X_copy[output_col] = 'No'

        X_copy.loc[
            (X_copy[cols[0]] == 1) | (X_copy[cols[1]] == 1) | 
            (X_copy[cols[2]] == 1) | (X_copy[cols[3]] == 1), output_col
        ] = 'Yes'

        return X_copy

class ComputingAverages(TransformerMixin, BaseEstimator):

    def __init__(self, output_col:str=None) -> None:
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

        X_copy[output_col] = 0

        for col in cols:
            temp = X_copy[col].apply(lambda x: self.encoder(x)).astype(float)

            X_copy[output_col] += temp

        X_copy[output_col] = X_copy[output_col]/len(cols)

        return X_copy
    
    @staticmethod
    def encoder(x:str)->float:

        if x is None: return np.nan
        else: return float(x.split(' - ')[0])

class ComputingRatio(TransformerMixin, BaseEstimator):

    def __init__(self, output_col) -> None:
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

        X_copy[output_col] = X_copy[cols[0]]/X_copy[cols[1]]

        return X_copy[output_col]

class Categorizer(TransformerMixin, BaseEstimator):

    def __init__(self, cutoffs:list, labels:dict, output_col:str, include_right:bool=True)->None:
        super().__init__()
        self.cutoffs   = cutoffs
        self.labels    = labels
        self.output_col= output_col
        self.include_right     = include_right

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()
        col   = X_copy.columns[0]

        output_col= self.output_col
        cutoffs   = self.cutoffs
        labels    = self.labels
        right = self.include_right

        bins = [-float('inf')] + cutoffs + [float('inf')]
        bins_labels = list(labels.keys())
        bins_labels.sort()

        X_copy[output_col] = pd.cut(X_copy[col], bins=bins, labels=bins_labels, right=right)

        X_copy[output_col] = X_copy[output_col].astype(float)

        X_copy[output_col] = X_copy[output_col].map(labels)

        return X_copy[output_col]

class RecodeGeoZone(TransformerMixin, BaseEstimator):

    def __init__(self, output_col) -> None:
        super().__init__()
        self.output_col = output_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()
        col = X_copy.columns

        output_col = self.output_col

        X_copy[output_col] = X_copy[col[0]].apply(lambda x: self.reencoder(x))

        return X_copy
    
    @staticmethod
    def reencoder(x:str)->str:

        if x is None: return None

        if x == 'Southern Zone': return 'Southern Zone'
        else: return 'Other Zone'

class CleanDatscan(BaseEstimator, TransformerMixin):

    def __init__(self, datscan_col:str, med_col:str) -> None:
        super().__init__()
        self.datscan_col = datscan_col
        self.med_col = med_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()
        datscan_col = self.datscan_col
        med_col = self.med_col

        X_copy.loc[X_copy[med_col] == 'Yes', datscan_col] = 'Yes'


        return X_copy
    
class CleanPramipexole(BaseEstimator, TransformerMixin):

    def __init__(self, pramipex_col:str, ropinerole_col:str) -> None:
        super().__init__()
        self.pramipex_col  = pramipex_col
        self.ropinerole_col= ropinerole_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy= X.copy()
        pramipex_col  = self.pramipex_col
        ropinerole_col= self.ropinerole_col

        X_copy.loc[X_copy[ropinerole_col] == 'Yes', pramipex_col] = 'Yes'


        return X_copy

class HandYcorrector(BaseEstimator, TransformerMixin):

    def __init__(self, status_col:str, ref_col:str='hyonoff') -> None:
        super().__init__()
        self.ref_col = ref_col
        self.status_col = status_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        cols = X_copy.columns

        ref_col = self.ref_col
        status_col = self.status_col

        X_copy[cols[2]] = X_copy.apply(
            lambda row: self.encoder(row[ref_col], row[cols[2]], row[status_col]), axis=1
        )
        
        return X_copy
    
    @staticmethod
    def encoder(reference:str, target:str, status)->str:

        if status == 'Control' or reference is None or target is None: return target

        if target == '0 - No signs of disease' and reference == 'Off':
            return '1 - Unilateral disease'

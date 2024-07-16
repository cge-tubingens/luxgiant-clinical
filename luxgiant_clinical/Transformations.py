"""
Module with transformers to process cleaned data from LuxGiant consortia
"""

from sklearn.base import TransformerMixin, BaseEstimator

import pandas as pd
import numpy as np

class Identity(BaseEstimator, TransformerMixin):

    """
    A scikit-learn transformer that performs an identity transformation.

    This transformer returns the input data unchanged. It can be used as a
    placeholder or in a pipeline where no transformation is required.
    """

    def __init__(self) -> None:
        """
        Initializes the Identity transformer.

        Parameters
        ----------
        None
        """
        super().__init__()

    def get_feature_names_out(self):
        """
        Get output feature names for transformation.

        Returns
        -------
        None
        """
        pass

    def fit(self, X:pd.DataFrame, y=None):
        """
        Fit the transformer on the input data.

        This method does nothing and is included to comply with the scikit-learn
        transformer interface.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : Identity
            Returns self.
        """
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        """
        Transform the input data by returning a copy of it.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_copy : pd.DataFrame
            A copy of the input data.
        """

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

class Subtype(TransformerMixin, BaseEstimator):

    def __init__(self, output_col:str, num_col:str, den_col:str) -> None:
        super().__init__()
        self.output_col= output_col
        self.num_col   = num_col
        self.den_col   = den_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()

        output_col = self.output_col
        num_col =self.num_col
        den_col =self.den_col

        X_copy['ratio'] = X_copy[num_col]/X_copy[den_col]

        mask_ratio_p    = (X_copy['ratio'] > 0)
        mask_rat_less_1 = (X_copy['ratio'] <= 1)
        mask_rat_grt_1  = (X_copy['ratio'] > 1)
        mask_rat_less_15= (X_copy['ratio'] < 1.5)
        mask_rat_grt_15 = (X_copy['ratio'] >= 1.5)

        X_copy.loc[mask_ratio_p & mask_rat_less_1, output_col]   = "Postural instability and gait difficulty"
        X_copy.loc[mask_rat_grt_1 & mask_rat_less_15, output_col]= "Indeterminate"
        X_copy.loc[mask_rat_grt_15, output_col]                  = "Tremor Dominant"

        mask_num_0 = (X_copy[num_col]==0)
        mask_num_p = (X_copy[num_col]>0)

        mask_den_0 = (X_copy[den_col]==0)
        mask_den_p = (X_copy[den_col]>0)

        X_copy.loc[mask_num_p & mask_den_0, output_col] = "Tremor Dominant"
        X_copy.loc[mask_num_0 & mask_den_p, output_col] = "Postural instability and gait difficulty"
        X_copy.loc[mask_num_0 & mask_den_0, output_col] = "Indeterminate"

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

        mask_patients = (X_copy[status_col] == 'Patient')

        reference = X_copy[ref_col].apply(
            lambda x: True if x is not None and x=='Off' else False
        ).astype(bool)

        target = X_copy[cols[2]].apply(
            lambda x: True if x is not None and x=='0 - No signs of disease' else False
        )
        target = target.fillna(False).astype(bool)

        mask = (mask_patients & reference & target)

        X_copy.loc[mask, cols[2]] = '1 - Unilateral disease'
        
        return X_copy

class AgeCategory(BaseEstimator, TransformerMixin):

    def __init__(self, status_col:str, onset_col:str, age_col:str) -> None:
        super().__init__()
        self.status_col = status_col
        self.age_col = age_col
        self.onset_col = onset_col

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col = X_copy.columns[0]

        status_col= self.status_col
        age_col   = self.age_col
        onset_col = self.onset_col

        mask_control = (X_copy[status_col]=='Control')

        X_copy.loc[mask_control, onset_col] = X_copy.loc[mask_control, age_col].copy()

        mask_group0 = (X_copy[onset_col]<21)

        mask_group1 = ((X_copy[onset_col]>=21) & (X_copy[onset_col]<50))

        mask_group2 = ((X_copy[onset_col]>=50)&(X_copy[onset_col]<=60))

        mask_group3 = (X_copy[onset_col]>60)

        X_copy.loc[mask_group0, 'age_category'] = "Onset <21 years"
        X_copy.loc[mask_group1, 'age_category'] = "Onset 21-49 years"
        X_copy.loc[mask_group2, 'age_category'] = "Onset 50-60 years"
        X_copy.loc[mask_group3, 'age_category'] = "Onset >60 years"

        return X_copy

import os

import pandas as pd
from pandas.io.stata import StataReader

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from luxgiant_clinical.ClinicalReport import contingency_table
from luxgiant_clinical.Transformations import AgeCategory, InitialMotorSymptoms, HandYOnOff
from luxgiant_clinical.Helpers import recover_columns_names

# DATA_MANAS= '/mnt/0A2AAC152AABFBB7/data/LuxGiantMatched/AGESEXMATCHED_GAPINDIA_DATA_23.01.2024.dta'
DATA_LUIS = '/mnt/0A2AAC152AABFBB7/data/RedCapLuxGiant/selected_data.dat'

# stata_manas= StataReader(DATA_MANAS)
stata_luis = StataReader(DATA_LUIS)

# df_manas = stata_manas.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)
df_luis  = stata_luis.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)

age_pipe1 = Pipeline([
    ('age1', AgeCategory(age_cutoff=50, output_col='AgeCat_1').set_output(transform='pandas'))
])
age_pipe2 = Pipeline([
    ('age2', AgeCategory(age_cutoff=40, output_col='AgeCat_2').set_output(transform='pandas'))
])
initmotor_pipe = Pipeline([
    ('initmotor', InitialMotorSymptoms(output_col='inmotnonmot').set_output(transform='pandas'))
])
onoff_pipe = Pipeline([
    ('hyonoff', HandYOnOff(output_col='hyonoff').set_output(transform='pandas'))
])

adv_trns = ColumnTransformer([
    ('age_pipe1', age_pipe1, ['age_at_onset']),
    ('age_pipe2', age_pipe2, ['age_at_onset']),
    ('initmotor_pipe', initmotor_pipe, ['initial_symptom_s___1', 'initial_symptom_s___2']),
    ('handyonoff', onoff_pipe, ['on_off', 'b_if_the_patient_is_receiv'])
],
remainder='passthrough').set_output(transform='pandas')

df_1 = adv_trns.fit_transform(df_luis)
df_1.columns = recover_columns_names(df_1.columns)
df_1

cont_region_status = contingency_table(
    df_luis, 'sites', 'Status',
    os.path.join('/mnt/0A2AAC152AABFBB7/CGE/luxgiant-clinical/results', 'cont_region_status')
)
cont_region_status


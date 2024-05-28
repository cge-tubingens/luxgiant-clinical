import os

import pandas as pd
from pandas.io.stata import StataReader

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from luxgiant_clinical.ClinicalReport import contingency_table
from luxgiant_clinical.Transformations import AgeCategory, InitialMotorSymptoms, HandYOnOff, HandYstage, ExposurePesticide, ComputingAverages, ComputingRatio
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
hystage_pipe = Pipeline()

adv_trns = ColumnTransformer([
    ('age_pipe1', age_pipe1, ['age_at_onset']),
    ('age_pipe2', age_pipe2, ['age_at_onset']),
    ('initmotor_pipe', initmotor_pipe, ['initial_symptom_s___1', 'initial_symptom_s___2']),
    ('handyonoff', onoff_pipe, ['on_off', 'b_if_the_patient_is_receiv']),
    ('handystages', HandYstage(output_col='hystage').set_output(transform='pandas'), ['hoehn_and_yahr_staging']),
    ('expopest', ExposurePesticide(output_col='exppesticide').set_output(transform='pandas'),[ 'nature_of_work___1', 'nature_of_work___2', 'over_your_lifetime_have_yo', 'during_your_lifetime_did_y'] ),
    ('num', ComputingAverages(output_col='num').set_output(transform='pandas'), ['tremor_7', 'tremor_at_rest_head_upper', 'tremor_at_rest_head_upper_4', 'tremor_at_rest_head_upper_2','tremor_at_rest_head_upper_3', 'tremor_at_rest_head_upper_5', 'action_or_postural_tremor', 'action_or_postural_tremor_2']),
    ('dem', ComputingAverages(output_col='dem').set_output(transform='pandas'), ['falling', 'freezing_when_walking', 'walking', 'gait', 'postural_stability_respons'])
],
remainder='passthrough').set_output(transform='pandas')

df_1 = adv_trns.fit_transform(df_luis)
df_1.columns = recover_columns_names(df_1.columns)
df_1

scnd_trns = ColumnTransformer([
    ('ratio', ComputingRatio(output_col='ratio').set_output(transform='pandas'), ['num', 'dem'])
],
remainder='passthrough').set_output(transform='pandas')

df_2 = scnd_trns.fit_transform(df_1)
df_2.columns = recover_columns_names(df_2.columns)
df_2

#cont_region_status = contingency_table(
#    df_luis, 'sites', 'Status',
#    os.path.join('/mnt/0A2AAC152AABFBB7/CGE/luxgiant-clinical/results', 'cont_region_status')
#)
#cont_region_status

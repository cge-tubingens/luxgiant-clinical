import os

import pandas as pd
from pandas.io.stata import StataReader

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from luxgiant_clinical.ClinicalReport import contingency_table
from luxgiant_clinical.Transformations import InitialMotorSymptoms, HandYOnOff, HandYstage, ExposurePesticide, ComputingAverages, ComputingRatio, Categorizer
from luxgiant_clinical.Helpers import recover_columns_names

# DATA_MANAS= '/mnt/0A2AAC152AABFBB7/data/LuxGiantMatched/AGESEXMATCHED_GAPINDIA_DATA_23.01.2024.dta'
DATA_LUIS = '/mnt/0A2AAC152AABFBB7/data/RedCapLuxGiant/selected_data.dat'

# stata_manas= StataReader(DATA_MANAS)
stata_luis = StataReader(DATA_LUIS)

# df_manas = stata_manas.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)
df_luis  = stata_luis.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)

age1_labels = {0: "<50", 1:">=50"}
age2_labels = {0: "<40", 1:">=40"}

adv_trns = ColumnTransformer([
    ('age_pipe1', Categorizer(cutoffs=[50], labels=age1_labels, output_col='agecat_1', include_right=False), ['age_at_onset']),
    ('age_pipe2', Categorizer(cutoffs=[40], labels=age1_labels, output_col='agecat_2', include_right=False), ['age_at_onset']),
    ('initmotor_pipe', InitialMotorSymptoms(output_col='inmotnonmot').set_output(transform='pandas'), ['initial_symptom_s___1', 'initial_symptom_s___2']),
    ('handyonoff', HandYOnOff(output_col='hyonoff').set_output(transform='pandas'), ['on_off', 'b_if_the_patient_is_receiv']),
    ('handystages', HandYstage(output_col='hystage').set_output(transform='pandas'), ['hoehn_and_yahr_staging']),
    ('expopest', ExposurePesticide(output_col='exppesticide').set_output(transform='pandas'),[ 'nature_of_work___1', 'nature_of_work___2', 'over_your_lifetime_have_yo', 'during_your_lifetime_did_y'] ),
    ('num', ComputingAverages(output_col='num').set_output(transform='pandas'), ['tremor_7', 'tremor_at_rest_head_upper', 'tremor_at_rest_head_upper_4', 'tremor_at_rest_head_upper_2','tremor_at_rest_head_upper_3', 'tremor_at_rest_head_upper_5', 'action_or_postural_tremor', 'action_or_postural_tremor_2']),
    ('dem', ComputingAverages(output_col='dem').set_output(transform='pandas'), ['falling', 'freezing_when_walking', 'walking', 'gait', 'postural_stability_respons'])
],
remainder='passthrough').set_output(transform='pandas')

df_1 = adv_trns.fit_transform(df_luis)
df_1.columns = recover_columns_names(df_1.columns)
df_1

ratio_labels = {1:"Tremor Dominant", 2:"Indeterminate", 3:"Postural instability and gait difficulty"}

ratio_pipe = Pipeline([
    ('ratio', ComputingRatio(output_col='ratio').set_output(transform='pandas')),
    ('ratio_cat', Categorizer(cutoffs=[1, 1.5], labels=ratio_labels, output_col='subtype', include_right=True))
])

scnd_trns = ColumnTransformer([
    ('ratio_pipe', ratio_pipe, ['num', 'dem'])
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

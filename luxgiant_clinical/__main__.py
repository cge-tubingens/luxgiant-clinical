"""
Command Line Interface
"""

import os

import pandas as pd

from pandas.io.stata import StataReader
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Helpers import arg_parser

def final_cleaning(df_data:pd.DataFrame)->pd.DataFrame:

    from luxgiant_clinical.Transformations import InitialMotorSymptoms, HandYOnOff, HandYstage, ExposurePesticide, ComputingAverages, ComputingRatio, Categorizer, RecodeGeoZone
    from luxgiant_clinical.Transformations import CleanDatscan, CleanPramipexole
    from luxgiant_clinical.Helpers import recover_columns_names

    age1_labels = {0: "<50", 1:">=50"}
    age2_labels = {0: "<40", 1:">=40"}
    age3_labels = {0:"<21", 1: "21-49", 2:"50-60", 3:">60"}
    age4_labels = {1:"<=30", 2: "31-40", 3:"41-50", 4:"51-60", 5:"61-70", 6:"71-80", 7:">80"}
    edu_labels  = {1:"<=12", 2:"Above 12"}

    adv_trns = ColumnTransformer([
        ('age_pipe1', Categorizer(cutoffs=[50], labels=age1_labels, output_col='agecat_1', include_right=False), ['age_at_onset']),
        ('age_pipe2', Categorizer(cutoffs=[40], labels=age2_labels, output_col='agecat_2', include_right=False), ['age_at_onset']),
        ('age_pipe3', Categorizer(cutoffs=[21, 50, 60], labels=age3_labels, output_col='agecatmd', include_right=False), ['age_at_onset']),
        ('age_pipe4', Categorizer(cutoffs=[30, 40, 50, 60, 70, 80], labels=age4_labels, output_col='agecent', include_right=True), ['age_at_onset']),
        ('edu_cat', Categorizer(cutoffs=[12], labels=edu_labels, output_col='education2', include_right=True), ['years_of_education']),
        ('initmotor_pipe', InitialMotorSymptoms(output_col='inmotnonmot').set_output(transform='pandas'), ['initial_symptom_s___1', 'initial_symptom_s___2']),
        ('handyonoff', HandYOnOff(output_col='hyonoff').set_output(transform='pandas'), ['on_off', 'b_if_the_patient_is_receiv']),
        ('handystages', HandYstage(output_col='hystage').set_output(transform='pandas'), ['hoehn_and_yahr_staging']),
        ('expopest', ExposurePesticide(output_col='exppesticide').set_output(transform='pandas'),[ 'nature_of_work___1', 'nature_of_work___2', 'over_your_lifetime_have_yo', 'during_your_lifetime_did_y'] ),
        ('num', ComputingAverages(output_col='num').set_output(transform='pandas'), ['tremor_7', 'tremor_at_rest_head_upper', 'tremor_at_rest_head_upper_4', 'tremor_at_rest_head_upper_2','tremor_at_rest_head_upper_3', 'tremor_at_rest_head_upper_5', 'action_or_postural_tremor', 'action_or_postural_tremor_2']),
        ('dem', ComputingAverages(output_col='dem').set_output(transform='pandas'), ['falling', 'freezing_when_walking', 'walking', 'gait', 'postural_stability_respons']),
        ('recodegeo', RecodeGeoZone(output_col='zonecat').set_output(transform='pandas'), ['zone_of_origin']),
        ('fix_datscan', CleanDatscan(datscan_col='datscan', med_col='f_dopa_pet').set_output(transform='pandas'), ['datscan', 'f_dopa_pet']),
        ('pramipexole', CleanPramipexole(pramipex_col='pramipexole', ropinerole_col='ropinerole'), ['pramipexole', 'ropinerole'])
    ],
    remainder='passthrough').set_output(transform='pandas')
    
    df_1 = adv_trns.fit_transform(df_data)
    df_1.columns = recover_columns_names(df_1.columns)
    
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
    
    return df_2

def execute_main()->None:

    args = arg_parser()
    args_dict = vars(args)

    input_file   = args_dict['input_file']
    output_folder= args_dict['output_folder']

    # check paths
    if not os.path.exists(input_file):
        raise FileNotFoundError("Input file cannot be found.")
    
    if not os.path.exists(output_folder):
        raise FileNotFoundError("Output folder cannot be found.")
    
    stata_data = StataReader(input_file)

    df  = stata_data.read(preserve_dtypes=False, convert_categoricals=True, convert_dates=True)

    df_clean = final_cleaning(df)

    del df

    df_clean.to_csv(os.path.join(output_folder, 'cleaned_file.csv'), index=False)

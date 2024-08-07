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

    from Transformations import InitialMotorSymptoms, HandYOnOff, HandYstage, ExposurePesticide, ComputingAverages, Subtype, Categorizer, RecodeGeoZone
    from Transformations import CleanDatscan, CleanPramipexole, Identity, HandYcorrector, AgeCategory
    from Helpers import recover_columns_names

    age1_labels = {0: "<50", 1:">=50"}
    age2_labels = {0: "<40", 1:">=40"}
    age4_labels = {1:"<=30", 2: "31-40", 3:"41-50", 4:"51-60", 5:"61-70", 6:"71-80", 7:">80"}
    edu_labels  = {1:"<=12", 2:"Above 12"}
    pdsl_labels = {0:"<=5", 1:">5"}

    adv_trns = ColumnTransformer([
        ('identity', Identity().set_output(transform='pandas'), ['age_at_onset', 'years_of_education', 'PD_duration']),
        ('pd_dur', Categorizer(cutoffs=[5], labels=pdsl_labels, output_col='pdsl', include_right=True), ['PD_duration']),
        ('age_pipe1', Categorizer(cutoffs=[50], labels=age1_labels, output_col='agecat_1', include_right=False), ['age_at_onset']),
        ('age_pipe2', Categorizer(cutoffs=[40], labels=age2_labels, output_col='agecat_2', include_right=False), ['age_at_onset']),
        ('age_pipe3', AgeCategory(status_col='Status', age_col='age', onset_col='age_at_onset').set_output(transform='pandas'), ['Status','age_at_onset', 'age']),
        ('age_pipe4', Categorizer(cutoffs=[30, 40, 50, 60, 70, 80], labels=age4_labels, output_col='agecent', include_right=True), ['age_at_onset']),
        ('edu_cat', Categorizer(cutoffs=[12], labels=edu_labels, output_col='education2', include_right=True), ['years_of_education']),
        ('initmotor_pipe', InitialMotorSymptoms(output_col='inmotnonmot').set_output(transform='pandas'), ['initial_symptom_s___1', 'initial_symptom_s___2']),
        ('handyonoff', HandYOnOff(output_col='hyonoff').set_output(transform='pandas'), ['on_off', 'b_if_the_patient_is_receiv']),
        ('handystages', HandYstage(output_col='hystage').set_output(transform='pandas'), ['hoehn_and_yahr_staging']),
        ('expopest', ExposurePesticide(output_col='exppesticide').set_output(transform='pandas'),[ 'nature_of_work___1', 'nature_of_work___2', 'over_your_lifetime_have_yo', 'during_your_lifetime_did_y'] ),
        ('num', ComputingAverages(output_col='num').set_output(transform='pandas'), ['tremor_7', 'tremor_at_rest_head_upper', 'tremor_at_rest_head_upper_4', 'tremor_at_rest_head_upper_2','tremor_at_rest_head_upper_3', 'tremor_at_rest_head_upper_5', 'action_or_postural_tremor', 'action_or_postural_tremor_2']),
        ('dem', ComputingAverages(output_col='den').set_output(transform='pandas'), ['falling', 'freezing_when_walking', 'walking', 'gait', 'postural_stability_respons']),
        ('recodegeo', RecodeGeoZone(output_col='zonecat').set_output(transform='pandas'), ['zone_of_origin']),
        ('fix_datscan', CleanDatscan(datscan_col='datscan', med_col='f_dopa_pet').set_output(transform='pandas'), ['datscan', 'f_dopa_pet']),
        ('pramipexole', CleanPramipexole(pramipex_col='pramipexole', ropinerole_col='ropinerole'), ['pramipexole', 'ropinerole'])
    ],
    remainder='passthrough').set_output(transform='pandas')
    
    df_1 = adv_trns.fit_transform(df_data)
    df_1.columns = recover_columns_names(df_1.columns)

    
    scnd_trns = ColumnTransformer([
        ('subtype', Subtype(output_col='subtype', num_col='num', den_col='den'), ['num', 'den']),
        ('handystage', HandYcorrector(ref_col='hyonoff', status_col='Status').set_output(transform='pandas'), ['hyonoff', 'Status', 'hoehn_and_yahr_staging'])
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

if __name__=="__main__":
    execute_main()

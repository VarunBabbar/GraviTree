# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:25:30 2023

@author: mn46
"""

import os
import numpy as np
import pandas as pd

from calc_sofa import calc_sofa

DATA_PATH = os.path.join('~', 'research', 'falling-models', 'data', 'MGH')

#################################################################
# Example for worst values in first 24 hours following admission
#################################################################

# def time_window(l):
#     # 24 hours
#     return l[l['Time_after_admission'].astype(str).apply(lambda x: x.startswith('0 days'))]

# Define features and their corresponding columns in the data, and how to aggregate them for SOFA calculation
# SOFA uses the worst value in the last 24 hours, which is defined differently for different features
feature_info = dict()
feature_info['bilirubin'] = {'numeric': ['Total_Bilirubin_mg_per_dL'], 'sofa_groupby': 'max'}
feature_info['bp_admission'] = {'numeric': ['SBP', 'DBP', 'MAP'], 'sofa_groupby': 'min'}
feature_info['creatinine'] = {'numeric': ['Creatinine_mg_per_dL'], 'sofa_groupby': 'max'}
feature_info['dobu_admission'] = {'numeric': ['Dobutamine_any_dose'], 'sofa_groupby': 'first'}
feature_info['dopamine'] = {'numeric': ['Dopamine_μg_kg_min'], 'sofa_groupby': 'max'}
feature_info['epinephrine'] = {'numeric': ['Epinephrine_μg_kg_min'], 'sofa_groupby': 'max'}
feature_info['gcs_admission'] = {'numeric': ['GCS'], 'sofa_groupby': 'min'}
feature_info['mech_vent_cpap_admission'] = {'numeric': ['mech_vent_cpap'], 'sofa_groupby': 'first'}
feature_info['norepinephrine'] = {'numeric': ['Norepinephrine_μg_kg_min'], 'sofa_groupby': 'max'}
feature_info['pao2fio2'] = {'numeric': ['PaO2', 'FiO2', 'PaO2_FiO2'], 'sofa_groupby': 'min'}
feature_info['platelets'] = {'numeric': ['Platelets_10_3_per_ml'], 'sofa_groupby': 'min'}
feature_info['uo_admission'] = {'numeric': ['Urine_output_mL'], 'sofa_groupby': 'sum'}

# list of all of the features used in the SOFA calculation
SOFA_FEATURES = [
    'Total_Bilirubin_mg_per_dL', 'MAP', 'Creatinine_mg_per_dL', 'Dobutamine_any_dose', 'Dopamine_μg_kg_min',
    'Epinephrine_μg_kg_min', 'GCS', 'mech_vent_cpap', 'Norepinephrine_μg_kg_min', 'PaO2_FiO2', 'Platelets_10_3_per_ml',
    'Urine_output_mL'
]


def generic_load_diagnostic_data(filename):
    # load raw data, filter to first 24 hours, index on Admission ID, and aggregate
    # according to the SOFA requirements defined above
    data = pd.read_csv(os.path.join(DATA_PATH, 'raw_data', filename + '.csv'))
    data['Time_after_admission'] = pd.to_timedelta(data['Time_after_admission'])
    data = data.loc[(data['Time_after_admission'] < pd.Timedelta('1 days'))]
    data = data.loc[:, ['Admission'] + feature_info[filename]['numeric']]
    data = data.groupby('Admission').agg(feature_info[filename]['sofa_groupby'])
    return data


if __name__ == '__main__':
    demographics = pd.read_csv(os.path.join(DATA_PATH, 'raw_data', 'admissions_demographics.csv'), index_col=0)
    demographics['Time_to_death'] = pd.to_timedelta(demographics['Time_to_death'])
    demographics['LOS_date_time'] = pd.to_timedelta(demographics['LOS_date_time'])
    demographics['LOS_date_based'] = pd.to_timedelta(demographics['LOS_date_based'])

    diagnostic_dfs = {filename: generic_load_diagnostic_data(filename) for filename in feature_info.keys()}

    full_patient_df = demographics.copy()
    for df_name in diagnostic_dfs:
        # select only numeric features from each diagnostic dataframe
        diagnostic = diagnostic_dfs[df_name]
        full_patient_df = full_patient_df.merge(diagnostic, on='Admission', how='outer')

    # add a column indicating whether the patient died in the ICU
    full_patient_df['died_in_ICU'] = (full_patient_df['Time_to_death'] <= full_patient_df['LOS_date_time'])

    print(len(full_patient_df))

    # get rid of top or bottom percentiles for some columns, b/c they're probably not real.
    # discard_99th = [
    #     "PaO2_FiO2", "Total_Bilirubin_mg_per_dL", "Platelets_10_3_per_ml", "Creatinine_mg_per_dL", "Urine_output_mL",
    #     "Dopamine_μg_kg_min", "Epinephrine_μg_kg_min", "Norepinephrine_μg_kg_min"
    # ]
    # discard_1st = ["PaO2_FiO2", "Platelets_10_3_per_ml", "Creatinine_mg_per_dL", "Urine_output_mL"]
    # top_percents = full_patient_df[discard_99th].quantile(0.99)
    # bott_percents = full_patient_df[discard_1st].quantile(0.01)

    # full_patient_df.loc[:, discard_99th] = \
    #     full_patient_df[discard_99th].where(full_patient_df[discard_99th] <= top_percents, np.nan)
    # full_patient_df.loc[:, discard_1st] = \
    #     full_patient_df[discard_1st].where(full_patient_df[discard_1st] >= bott_percents, np.nan)

    # replace nan with 0 for these meds
    fill_na_cols = [
        "Dobutamine_any_dose", "Dopamine_μg_kg_min", "Epinephrine_μg_kg_min", "Norepinephrine_μg_kg_min",
        "mech_vent_cpap"
    ]
    full_patient_df.loc[:, fill_na_cols] = full_patient_df[fill_na_cols].fillna(0)

    full_patient_df.to_csv(os.path.join(DATA_PATH, 'processed_data', 'full_patient_df.csv'))

    # keep only patients for which all SOFA features are available
    sofa_df = full_patient_df.dropna(axis=0, subset=SOFA_FEATURES)

    print(len(sofa_df))

    # calculate SOFA scores
    real_sofas = sofa_df.apply(calc_sofa, axis=1)
    print(f"Number of patients with a valid SOFA score: {len(real_sofas)}")

    # print(real_sofas)

    died_within_ = (sofa_df['Time_to_death'] < pd.Timedelta('1 days'))
    print(f"Number of patients who died in the time window: {died_within_.sum()}")

    print(
        f"Mean SOFA score of patients who died in the time window: {real_sofas[died_within_].mean():0.2f} +/- {real_sofas[died_within_].std():0.2f}"
    )
    print(
        f"Mean SOFA score of patients who did not die in the time window: {real_sofas[~died_within_].mean():0.2f} +/- {real_sofas[~died_within_].std():0.2f}"
    )
    print(
        f"Mean SOFA score of patients who died in the ICU: {real_sofas[sofa_df['died_in_ICU']].mean():0.2f} +/- {real_sofas[sofa_df['died_in_ICU']].std():0.2f}"
    )

    sofa_df['sofa_score'] = real_sofas
    sofa_df['died_within_24h'] = died_within_
    sofa_df.to_csv(os.path.join(DATA_PATH, 'processed_data', 'sofa_scores.csv'))

# if __name__ == '__main__':
#     ################
#     # Load data
#     ################

#     df = pd.read_csv(os.path.join(path0, 'admissions_demographics.csv'))

#     mech_vent_cpap = pd.read_csv(os.path.join(path0, 'mech_vent_cpap_admission.csv'))
#     mech_vent_cpap = time_window(mech_vent_cpap)

#     uo = pd.read_csv(os.path.join(path0, 'uo_admission.csv'))
#     uo = time_window(uo)
#     uo = uo.groupby(['Admission'])['Urine_output_mL'].sum().reset_index()

#     gcs = pd.read_csv(os.path.join(path0, 'gcs_admission.csv'))
#     gcs = time_window(gcs)
#     # worst value
#     gcs = gcs.groupby(['Admission'])['GCS'].min().reset_index()

#     bp = pd.read_csv(os.path.join(path0, 'bp_admission.csv'))
#     bp = time_window(bp)
#     # worst value
#     bp = bp.groupby(['Admission'])['MAP'].min().reset_index()

#     bili = pd.read_csv(os.path.join(path0, 'bilirubin.csv'))
#     bili = time_window(bili)
#     # worst value
#     bili = bili.groupby(['Admission'])['Total_Bilirubin_mg_per_dL'].max().reset_index()

#     plt = pd.read_csv(os.path.join(path0, 'platelets.csv'))
#     plt = time_window(plt)
#     # worst value
#     plt = plt.groupby(['Admission'])['Platelets_10_3_per_ml'].min().reset_index()

#     creat = pd.read_csv(os.path.join(path0, 'creatinine.csv'))
#     creat = time_window(creat)
#     # worst value
#     creat = creat.groupby(['Admission'])['Creatinine_mg_per_dL'].max().reset_index()

#     pao2fio2 = pd.read_csv(os.path.join(path0, 'pao2fio2.csv'))
#     pao2fio2 = time_window(pao2fio2)
#     # worst value (PaO2/FiO2)
#     pao2fio2 = pao2fio2.groupby(['Admission'])['PaO2_FiO2'].min().reset_index()

#     dobu = pd.read_csv(os.path.join(path0, 'dobu_admission.csv'))
#     dobu = time_window(dobu)

#     # worst values = highest dose
#     dopa = pd.read_csv(os.path.join(path0, 'dopamine.csv'))
#     dopa = time_window(dopa)
#     # worst value
#     dopa = dopa.groupby(['Admission'])['Dopamine_μg_kg_min'].max().reset_index()

#     epi = pd.read_csv(os.path.join(path0, 'epinephrine.csv'))
#     epi = time_window(epi)
#     # worst value
#     epi = epi.groupby(['Admission'])['Epinephrine_μg_kg_min'].max().reset_index()

#     nore = pd.read_csv(os.path.join(path0, 'norepinephrine.csv'))
#     nore = time_window(nore)
#     # worst value
#     nore = nore.groupby(['Admission'])['Norepinephrine_μg_kg_min'].max().reset_index()

#     # Join all

#     df = pd.merge(df, mech_vent_cpap, on='Admission', how='outer')
#     df = pd.merge(df, uo, on='Admission', how='outer')
#     df = pd.merge(df, gcs, on='Admission', how='outer')
#     df = pd.merge(df, bp, on='Admission', how='outer')
#     df = pd.merge(df, bili, on='Admission', how='outer')
#     df = pd.merge(df, plt, on='Admission', how='outer')
#     df = pd.merge(df, creat, on='Admission', how='outer')
#     df = pd.merge(df, pao2fio2, on='Admission', how='outer')
#     df = pd.merge(df, dobu, on='Admission', how='outer')
#     df = pd.merge(df, dopa, on='Admission', how='outer')
#     df = pd.merge(df, epi, on='Admission', how='outer')
#     df = pd.merge(df, nore, on='Admission', how='outer')

#     df = df.drop(columns=['Time_after_admission_x', 'Time_after_admission_y']).drop_duplicates()

#     df.loc[df['mech_vent_cpap'] != 1, 'mech_vent_cpap'] = 0
#     df.loc[df['Dobutamine_any_dose'] != 1, 'Dobutamine_any_dose'] = 0
#     cs = ['Dopamine_μg_kg_min', 'Epinephrine_μg_kg_min', 'Norepinephrine_μg_kg_min']

#     df[cs] = df[cs].fillna(0)
#     df = df.reset_index(drop=True)

#     # Calculate Sofa scores for data with no missing values, but possibly outliers

#     comp = df[[
#         'mech_vent_cpap', 'Urine_output_mL', 'GCS', 'MAP', 'Total_Bilirubin_mg_per_dL', 'Platelets_10_3_per_ml',
#         'Creatinine_mg_per_dL', 'PaO2_FiO2', 'Dobutamine_any_dose', 'Dopamine_μg_kg_min', 'Epinephrine_μg_kg_min',
#         'Norepinephrine_μg_kg_min'
#     ]].dropna()

#     real_sofas_with_possibly_outliers = comp.apply(lambda x: calc_sofa(x), axis=1)

#     #######################
#     # Dealing with outliers
#     #######################

#     df1 = df[[
#         'SexDSC', 'GCS', 'PaO2_FiO2', 'mech_vent_cpap', 'Total_Bilirubin_mg_per_dL', 'Platelets_10_3_per_ml',
#         'Creatinine_mg_per_dL', 'Urine_output_mL', 'MAP', 'Dobutamine_any_dose', 'Dopamine_μg_kg_min',
#         'Epinephrine_μg_kg_min', 'Norepinephrine_μg_kg_min'
#     ]]

#     df1.loc[:, 'SexDSC'] = df1['SexDSC'].replace(['Female', 'Male'], ['1.', '-1.']).astype(float)

#     num_pts = df1.shape[0]
#     num_cols = df1.shape[1]
#     refNum = 10

#     # get rid of top or bottom percentiles for some columns, b/c they're probably not real.
#     discard_99th = [
#         "PaO2_FiO2", "Total_Bilirubin_mg_per_dL", "Platelets_10_3_per_ml", "Creatinine_mg_per_dL", "Urine_output_mL",
#         "Dopamine_μg_kg_min", "Epinephrine_μg_kg_min", "Norepinephrine_μg_kg_min"
#     ]
#     discard_1st = ["PaO2_FiO2", "Platelets_10_3_per_ml", "Creatinine_mg_per_dL", "Urine_output_mL"]
#     top_percents = df1[discard_99th].quantile(0.99)
#     bott_percents = df1[discard_1st].quantile(0.01)

#     df1.loc[:, discard_99th] = df1[discard_99th].where(df1[discard_99th] <= top_percents, np.nan)
#     df1.loc[:, discard_1st] = df1[discard_1st].where(df1[discard_1st] >= bott_percents, np.nan)

#     # replace nan with 0 for these meds

#     fill_na_cols = ["Dobutamine_any_dose", "Dopamine_μg_kg_min", "Epinephrine_μg_kg_min", "Norepinephrine_μg_kg_min"]
#     df1.loc[:, fill_na_cols] = df1[fill_na_cols].fillna(0)

#     ## make copy of original values
#     df0 = df1.copy()

#     #######################
#     # Normalization
#     #######################

#     ## z- normalize
#     means = df1.mean()
#     IQRs = df1.quantile(0.75) - df1.quantile(0.25)

#     binary_cols = ["Dobutamine_any_dose", "SexDSC"]
#     med_amt_cols = ["Dopamine_μg_kg_min", "Epinephrine_μg_kg_min", "Norepinephrine_μg_kg_min"]

#     # if it's binary, subtract the mean and divide by the range/2
#     df1_binary_cols = df1.loc[:, binary_cols + med_amt_cols]
#     df1.loc[:, df1_binary_cols.columns] = ((df1_binary_cols - means[df1_binary_cols.columns]) *
#                                            (2 / np.nanmax(df1_binary_cols) - np.nanmin(df1_binary_cols))).astype(float)

#     # if it's not binary, subtract the mean and divide by the IQR
#     df1_not_binary_cols = df1.loc[:, [col for col in df1.columns if col not in binary_cols + med_amt_cols]]
#     df1.loc[:, df1_not_binary_cols.columns] = ((df1_not_binary_cols - means[df1_not_binary_cols.columns]) /
#                                                IQRs[df1_not_binary_cols.columns]).astype(float)

#     #######################
#     # Remove nan components
#     #######################

#     df1.loc[:, med_amt_cols] = df1[med_amt_cols].fillna(0)

#     df2 = df1.dropna(axis=0)
#     df0 = df0.dropna(axis=0)

#     num_pts = df2.shape[0]
#     num_cols = df2.shape[1]

#     #######################
#     # Calculate sofa scores
#     #######################

#     real_sofas = df0.apply(calc_sofa, axis=1)
#     print(df0)
#     print()

#     print(real_sofas)

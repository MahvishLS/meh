import pandas as pd

df_vitals = pd.read_csv('patient_vitals.csv')
df_labs = pd.read_csv('lab_results.csv')

# Horizontal Merging
df_horizontal_merge = pd.merge(
    df_vitals,
    df_labs,
    on='Patient_ID',
    how='inner'
)

df_horizontal_merge.head(5)

# Vertical Merging
df_new_patients = df_labs[['Patient_ID']].drop_duplicates()
df_new_patients['Source'] = 'Lab Results'

df_existing_patients = df_vitals[['Patient_ID', 'Hospital_Location']].drop_duplicates()
df_existing_patients['Source'] = 'Vitals'

df_vertical_merge = pd.concat([df_existing_patients, df_new_patients], ignore_index=True)
df_vertical_merge.drop_duplicates(subset=['Patient_ID'], keep='first', inplace=True)

df_vertical_merge.head(5)

# Semantic Merging
diagnosis_mapping = {
    'HTN': '44054006 (Hypertension)',
    'DM-II': '73211009 (Diabetes Mellitus Type 2)',
    'Asthma': '195967001 (Asthma)',
    'Migr.': '59368008 (Migraine)'
}

df_vitals['Diagnosis_Code_Standard'] = df_vitals['Diagnosis_Code'].replace(diagnosis_mapping)

import numpy as np

medication_mapping = {
    'Lisi.': 'Lisinopril',
    'Metform.': 'Metformin',
    'Simbast.': 'Simvastatin',
    'N/A': np.nan
}

df_labs['Medication_Name_Standard'] = df_labs['Medication_Name'].replace(medication_mapping)

gender_mapping = {
    'F': 'Female',
    'M': 'Male'
}
df_vitals['Gender_Standard'] = df_vitals['Gender'].replace(gender_mapping)
df_vitals.head(5)



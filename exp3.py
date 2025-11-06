import pandas as pd
df = pd.read_csv('uncleaned_data.csv')

df.drop_duplicates(subset=df.columns.difference(['Patient_ID']))

# Drop irrelevant feature
df.drop('Admission_Date', axis=1, inplace=True)
df.rename(columns={'Lab_Results (Glucose)': 'Glucose_mg_dL'}, inplace=True)

import numpy as np

df.replace(['NA', 'NaN', 'Missing', ''], np.nan, inplace=True)
df.info()
# Conversions
df['Blood_Pressure'] = pd.to_numeric(df['Blood_Pressure'])
df['Glucose_mg_dL'] = df['Glucose_mg_dL'].astype(str).str.replace(' mg/dL', '', regex=False)
df['Glucose_mg_dL'] = pd.to_numeric(df['Glucose_mg_dL'])
df['Age_Years'].hist().plot()
df['Weight_kg'].hist().plot()

# Correct erroneous entries
df.loc[(df['Age_Years'] <= 0) | (df['Age_Years'] >= 110), 'Age_Years'] = np.nan
df.loc[(df['Weight_kg'] <= 0) | (df['Weight_kg'] >= 200), 'Weight_kg'] = np.nan

# Interpolation
median_imputer = df.median(numeric_only=True)
df.fillna(median_imputer).head(4)

mode_diagnosis = df['Diagnosis'].mode()[0]
df['Diagnosis'] = df['Diagnosis'].fillna(mode_diagnosis)

# Standardization
df['Gender'] = df['Gender'].replace({'F': 'Female', 'M': 'Male', 'Other': 'Unknown'})

df['Insurance_Type'].replace({'Govt': 'Public'})

df['Gender'] = df['Gender'].astype('category')
df['Insurance_Type'] = df['Insurance_Type'].astype('category')
df['Diagnosis'] = df['Diagnosis'].astype('category')


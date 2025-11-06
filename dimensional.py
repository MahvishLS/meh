import pandas as pd

df = pd.read_csv('breast_cancer.csv')

X = df.drop('target', axis=1)
Y = df['target']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

C = np.cov(X_scaled.T)
plt.figure(figsize=(20, 10))
sns.heatmap(C, annot=True)
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(C)
print(eigenvalues)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2']
)

pc_df['target'] = Y

plt.figure(figsize=(6, 3))
sns.scatterplot(
    data=pc_df,
    x='PC1',
    y='PC2',
    hue='target'
)


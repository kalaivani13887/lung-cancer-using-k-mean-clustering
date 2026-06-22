principle component analysis features reduce pandrathu atha etho da velayee prom data prepare pannuthu



import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# Load Data
df = pd.read_csv('survey lung cancer.csv')

print("Original Data Loaded:", df.shape)

# Data Cleaning
df = df.drop_duplicates()

df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

print("After Cleaning:", df.shape)

# Encoding
for c in df.select_dtypes(include=['object', 'string']):
    df[c] = LabelEncoder().fit_transform(df[c])

print("After Encoding:")
print(df.head())

# Remove target
x = df.drop('LUNG_CANCER', axis=1)

# Scale Data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print("After PCA Shape:", x_pca.shape)

# K-Means
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(x_pca)

# Save Cluster
df['Cluster'] = clusters

print("\nCluster Output:")
print(df[['Cluster']].head())

# Evaluate
score = silhouette_score(x_pca, clusters)

print("Silhouette Score:", round(score, 2))

# Save Model
joblib.dump(model, 'lung_cancer_kmeans_pca.pkl')
joblib.dump(pca, 'pca_model.pkl')

print("Model and PCA saved successfully!")/*ethula accuracy varathu yana ithu unsupervisied learinig

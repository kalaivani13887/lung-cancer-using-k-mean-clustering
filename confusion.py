import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Load Data
df = pd.read_csv('survey lung cancer.csv')

print("Original Data:", df.shape)

# Cleaning
df = df.drop_duplicates()

df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Encoding
for c in df.select_dtypes(include=['object', 'string']):
    df[c] = LabelEncoder().fit_transform(df[c])

# Actual Target
y = df['LUNG_CANCER']

# Features
x = df.drop('LUNG_CANCER', axis=1)

# Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# KMeans
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

pred = model.fit_predict(x_pca)

# Confusion Matrix
cm = confusion_matrix(y, pred)

print("Confusion Matrix:")
print(cm)

# Plot
plt.figure(figsize=(5,5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j])

plt.show()

# Save
joblib.dump(model,'kmeans_pca.pkl')
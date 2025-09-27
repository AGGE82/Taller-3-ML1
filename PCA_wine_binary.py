import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Load wine quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df1 = pd.read_csv(url, sep=';')

# Convert quality scores to three classes
def quality_to_class(quality):
    if quality <= 5:
        return 'poor'
    else:
        return 'good'

# Add new column with three classes
df1['quality_class'] = df1['quality'].apply(quality_to_class)

# Show distribution of new classes
print("Three-Class Distribution:\n", df1['quality_class'].value_counts())

# Visualize class distribution
plt.figure()
sns.countplot(data=df1, x='quality_class', order=['poor', 'good'])
plt.title('Wine Quality Class Distribution')
plt.show()

# Load and prepare the wine dataset
X = df1.drop(['quality','quality_class'], axis=1)
y = df1['quality_class']

# 1. Train baseline k-NN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline model without PCA
start_time = time.time()
baseline_model = KNeighborsClassifier(n_neighbors=3)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_time = time.time() - start_time

print("Baseline Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
print(f"Training time: {baseline_time:.4f} seconds\n")

pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

start_time = time.time()
pca_pipeline.fit(X_train, y_train)
pipeline_pred = pca_pipeline.predict(X_test)
pipeline_time = time.time() - start_time

print("PCA Pipeline Performance:")
print(f"Accuracy: {accuracy_score(y_test, pipeline_pred):.4f}")
print(f"Training time: {pipeline_time:.4f} seconds\n")

pca = PCA()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca.fit(X_scaled)

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

le = LabelEncoder()
y = le.fit_transform(y)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.xlabel(f'First Principal Component')
plt.ylabel(f'Second Principal Component')
plt.title('Wine Dataset - First Two Principal Components')
plt.colorbar(scatter)
plt.show()

# Print the explained variance ratio for the first two components
print("Explained variance ratio for first two components:")
print(f"PC1: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"Total: {sum(pca_2d.explained_variance_ratio_):.4f}")
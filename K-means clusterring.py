import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
file_path = "C:/Users/kkkos/Desktop/Datasets/Mall_Customers.csv"
df = pd.read_csv(file_path)

# Dataset preview
print("Dataset preview:")
print(df.head())

# Dataset info and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Data Preprocessing
# Drop non-numerical columns (like 'CustomerID', 'Gender') and handle categorical variables
df = df.drop(columns=['CustomerID', 'Gender'])

# Check for missing values after dropping columns (should be none)
print("\nMissing Values after drop:")
print(df.isnull().sum())

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Preview of the scaled data
print("\nScaled Data:")
print(pd.DataFrame(scaled_data, columns=df.columns).head())

# Step 4: Elbow Method to determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply KMeans with the optimal number of clusters (let's assume it's 5 after elbow method)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the dataframe
df['Cluster'] = y_kmeans

# Step 6: Visualizing the Clusters (2D plot)
plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 1], df.iloc[:, 0], c=df['Cluster'], cmap='viridis', marker='o')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Step 7: Cluster Profiling
# Calculate the average of each feature per cluster
cluster_profile = df.groupby('Cluster').mean()
print("\nCluster Profiling:")
print(cluster_profile)

# Step 8: Silhouette Score to evaluate clustering quality
silhouette_avg = silhouette_score(scaled_data, y_kmeans)
print(f"\nSilhouette Score: {silhouette_avg}")

# Step 9: Prediction for a new customer
new_customer = [[25, 50, 75]]  # Replace with new customer data (Age, Annual Income, Spending Score)
new_customer_scaled = scaler.transform(new_customer)
cluster_label = kmeans.predict(new_customer_scaled)
print(f"\nThe new customer belongs to Cluster: {cluster_label[0]}")

# Step 10: (Optional) 3D Plot to visualize data
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=y_kmeans, cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('Customer Segments (3D)')
plt.show()


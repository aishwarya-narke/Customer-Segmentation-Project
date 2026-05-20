import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# STEP 1: Load dataset
df = pd.read_csv(r"C:\Users\AISHWARYA NARKE\Desktop\Customer_Segmentation_Project\customers.csv")

# STEP 2: Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# STEP 3: Select features for clustering
X = df[['Annual Income(K$)', 'Spending Score (1-100)']]

# STEP 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# =========================
# 🔍 ANALYSIS SECTION
# =========================

print("\n--- Cluster Summary ---")
print(df.groupby('Cluster')[['Annual Income(K$)', 'Spending Score (1-100)']].mean())

# Customer count per cluster
print("\n--- Customers per Cluster ---")
print(df['Cluster'].value_counts())

# Gender distribution
print("\n--- Gender Distribution per Cluster ---")
print(pd.crosstab(df['Cluster'], df['Gender']))

# Age analysis
print("\n--- Average Age per Cluster ---")
print(df.groupby('Cluster')['Age'].mean())

# =========================
# 📊 VISUALIZATION SECTION
# =========================

# 1. Scatter Plot (Main segmentation)
plt.figure()
plt.scatter(df['Annual Income(K$)'], df['Spending Score (1-100)'], c=df['Cluster'])

# Cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

# 2. Cluster Size Bar Chart
df['Cluster'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# 3. Average Spending per Cluster
df.groupby('Cluster')['Spending Score (1-100)'].mean().plot(kind='bar')
plt.title("Average Spending Score per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Spending Score")
plt.show()
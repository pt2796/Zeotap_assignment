from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Load datasets
customers = pd.read_csv('data/Customers.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Create customer aggregation DataFrame
customer_agg = customers[['CustomerID']].copy()

# Aggregate transaction data
transaction_agg = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique'
}).reset_index()

transaction_agg.rename(columns={'ProductID': 'NumProductsPurchased'}, inplace=True)

# Merge to ensure all customers are included
customer_agg = customer_agg.merge(transaction_agg, on='CustomerID', how='left')

# Fill missing values with 0 for customers without transactions
customer_agg['TotalValue'] = customer_agg['TotalValue'].fillna(0)
customer_agg['Quantity'] = customer_agg['Quantity'].fillna(0)
customer_agg['NumProductsPurchased'] = customer_agg['NumProductsPurchased'].fillna(0)

# Normalize the data
scaler = StandardScaler()
features = ['TotalValue', 'Quantity', 'NumProductsPurchased']
customer_agg_scaled = scaler.fit_transform(customer_agg[features])

# Perform K-Means clustering
db_scores = []
for k in range(2, 11):  # Test between 2 and 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_agg_scaled)
    db_scores.append((k, davies_bouldin_score(customer_agg_scaled, kmeans.labels_)))

# Select the best number of clusters based on Davies-Bouldin Index
best_k = min(db_scores, key=lambda x: x[1])[0]
kmeans = KMeans(n_clusters=best_k, random_state=42)
customer_agg['Cluster'] = kmeans.fit_predict(customer_agg_scaled)

# Perform PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(customer_agg_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=customer_agg['Cluster'], palette='Set2')
plt.title('Customer Segments')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('outputs/customer_segments.png')
plt.close()

# Save the clustering results
customer_agg.to_csv('outputs/customer_clustering.csv', index=False)

# Print the best number of clusters
print(f"Optimal number of clusters: {best_k}")
print("Clustering results saved to outputs/customer_clustering.csv.")
print("Cluster visualization saved to outputs/customer_segments.png.")

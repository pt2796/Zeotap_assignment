from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load datasets
customers = pd.read_csv('data/Customers.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Merge customers with aggregated transaction data
customer_agg = customers[['CustomerID']].copy()

# Aggregate transaction data
transaction_agg = transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique'
}).reset_index()

transaction_agg.rename(columns={'ProductID': 'NumProductsPurchased'}, inplace=True)

# Merge with customer_agg to ensure all customers are included
customer_agg = customer_agg.merge(transaction_agg, on='CustomerID', how='left')

# Fill missing values with 0 for customers without transactions
customer_agg['TotalValue'] = customer_agg['TotalValue'].fillna(0)
customer_agg['Quantity'] = customer_agg['Quantity'].fillna(0)
customer_agg['NumProductsPurchased'] = customer_agg['NumProductsPurchased'].fillna(0)

# Normalize data
scaler = StandardScaler()
features = ['TotalValue', 'Quantity', 'NumProductsPurchased']
customer_agg_scaled = scaler.fit_transform(customer_agg[features])

# Compute cosine similarity
similarity_matrix = cosine_similarity(customer_agg_scaled)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_agg['CustomerID'], columns=customer_agg['CustomerID'])

# Recommendation function
def get_top_3_lookalikes(customer_id, similarity_df):
    try:
        similar_customers = similarity_df.loc[customer_id].sort_values(ascending=False).iloc[1:4]
        return similar_customers
    except KeyError:
        return pd.Series()

# Generate recommendations for the first 20 customers
recommendations = {}
for customer_id in customer_agg['CustomerID'].head(20):
    top_3 = get_top_3_lookalikes(customer_id, similarity_df)
    if not top_3.empty:
        recommendations[customer_id] = list(top_3.index)
    else:
        recommendations[customer_id] = ['N/A', 'N/A', 'N/A']

# Save recommendations to CSV
recommendations_df = pd.DataFrame.from_dict(
    recommendations, 
    orient='index', 
    columns=['Lookalike_1', 'Lookalike_2', 'Lookalike_3']
)
recommendations_df.reset_index(inplace=True)
recommendations_df.rename(columns={'index': 'CustomerID'}, inplace=True)
recommendations_df.to_csv('outputs/Lookalike.csv', index=False)

print("Lookalike recommendations saved to outputs/Lookalike.csv.")

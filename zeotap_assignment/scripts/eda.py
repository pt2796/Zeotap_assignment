import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('data/Customers.csv')
products = pd.read_csv('data/Products.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Check for missing values and duplicates
print("Missing values and duplicates:")
for df, name in zip([customers, products, transactions], ['Customers', 'Products', 'Transactions']):
    print(f"\n{name}:")
    print(df.isnull().sum())
    print(f"Duplicates: {df.duplicated().sum()}")

# Drop duplicates
customers.drop_duplicates(inplace=True)
products.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)

# Convert date columns to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Feature engineering
transactions['TotalValue'] = transactions['Quantity'] * transactions['Price']

# Visualizations
# 1. Distribution of Product Prices
plt.figure(figsize=(10, 6))
sns.histplot(products['Price'], kde=True, color='blue')
plt.title('Distribution of Product Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.savefig('outputs/product_price_distribution.png')
plt.close()

# 2. Sales Trends Over Time
monthly_sales = transactions.groupby(transactions['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', color='green')
plt.title('Total Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales (USD)')
plt.savefig('outputs/sales_trends.png')
plt.close()

# 3. Total Sales by Region
region_sales = customers.merge(transactions, on='CustomerID').groupby('Region')['TotalValue'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='TotalValue', data=region_sales, palette='viridis')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales (USD)')
plt.savefig('outputs/sales_by_region.png')
plt.close()

# 4. Top 10 Most Popular Products
popular_products = transactions.groupby('ProductID')['Quantity'].sum().reset_index()
popular_products = popular_products.merge(products[['ProductID', 'ProductName']], on='ProductID')
popular_products = popular_products.sort_values(by='Quantity', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Quantity', y='ProductName', data=popular_products.head(10), palette='coolwarm')
plt.title('Top 10 Most Popular Products')
plt.xlabel('Quantity Sold')
plt.ylabel('Product Name')
plt.savefig('outputs/top_products.png')
plt.close()

print("EDA visualizations saved in the outputs directory.")

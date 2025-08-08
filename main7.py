import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# ===================== STEP 1: DATA PREPARATION =====================
# Read the dataset from Excel
df = pd.read_excel("Sales_Data_Enhanced.xlsx")

# ===================== STEP 2: DATA PREPROCESSING =====================
# Normalize column names
df.columns = df.columns.str.strip().str.replace(" ", "")

# Convert OrderDate column to datetime format
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')

# Remove rows with missing values
df = df.dropna()

# Add a column for total sales for each transaction
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# ===================== STEP 3: DATA ANALYSIS =====================
# Descriptive analysis
total_orders = df['OrderID'].nunique()
total_customers = df['CustomerName'].nunique()
total_revenue = df['TotalSales'].sum()

# Top 5 customers by revenue
top_customers = df.groupby('CustomerName')['TotalSales'].sum().sort_values(ascending=False).head(5)

# Predicting next month's revenue using linear regression
df_monthly = df.groupby(pd.Grouper(key='OrderDate', freq='ME'))['TotalSales'].sum().reset_index()
df_monthly['Month_Num'] = np.arange(len(df_monthly))

X = df_monthly[['Month_Num']]
y = df_monthly['TotalSales']
model = LinearRegression()
model.fit(X, y)
next_month_pred = model.predict([[len(df_monthly)]])[0]

# Customer segmentation (clustering) based on purchase behavior
customer_summary = df.groupby('CustomerName').agg({'TotalSales': 'sum', 'OrderID': 'count'}).reset_index()
kmeans = KMeans(n_clusters=3, random_state=42)
customer_summary['Cluster'] = kmeans.fit_predict(customer_summary[['TotalSales', 'OrderID']])

# ===================== STEP 4: DATA VISUALIZATION =====================
# Top 10 products by revenue
top_products = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.show()

# Monthly revenue trend
monthly_sales = df.groupby(df['OrderDate'].dt.to_period('M'))['TotalSales'].sum().reset_index()
monthly_sales['OrderDate'] = monthly_sales['OrderDate'].dt.to_timestamp()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['OrderDate'], monthly_sales['TotalSales'], marker='o', color='orange')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# ===================== STEP 5: BUSINESS RECOMMENDATIONS =====================
recommendations = [
    "Reduce inventory for slow-moving products to avoid overstock.",
    "Consider removing products with low sales performance."
]

# ===================== OUTPUT RESULTS =====================
results = {
    "Total Orders": total_orders,
    "Total Customers": total_customers,
    "Total Revenue": total_revenue,
    "Top Customers": top_customers,
    "Predicted Next Month Sales": next_month_pred,
    "Business Recommendations": recommendations
}

print(results)
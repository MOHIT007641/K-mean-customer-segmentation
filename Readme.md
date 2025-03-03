Customer Segmentation using K-Means Clustering

📌 Overview

Customer segmentation is a key marketing strategy that divides customers into distinct groups based on common characteristics. This project utilizes K-Means Clustering to classify customers into different segments based on Annual Income and Spending Score.

🔍 Problem Statement

Businesses and retailers need to identify different customer segments to tailor marketing strategies effectively. However, manual segmentation is inefficient and lacks accuracy. This project aims to:

Automatically group customers based on their purchasing behavior.

Help businesses target specific customer groups for personalized promotions.

Enhance customer satisfaction and business revenue.

📊 Dataset Information

Dataset Name: Mall_Customers.csv

Features Used:

Annual Income (in $1000s)

Spending Score (1-100)

Target: Customer segmentation into different clusters.

🛠️ Steps Followed

1️⃣ Data Preprocessing

Load dataset using Pandas.

Check for missing values and data types.

Extract relevant features (Annual Income, Spending Score).

2️⃣ Finding Optimal Number of Clusters (Elbow Method)

Use Within-Cluster Sum of Squares (WCSS) to determine the best number of clusters.

Plot an Elbow Graph to visualize the optimal cluster count.

3️⃣ Applying K-Means Clustering

Implement K-Means with the optimal cluster count (K=5).

Assign cluster labels to each customer.

4️⃣ Data Visualization

Scatter plot to visualize customer clusters based on Annual Income vs. Spending Score.

Mark centroids of each cluster.

📌 Implementation Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Dataset
customer_data = pd.read_csv('/content/Mall_Customers.csv')

# Selecting features
X = customer_data.iloc[:, [3, 4]].values

# Finding optimal clusters using WCSS
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
![Elbow_Finding](https://github.com/user-attachments/assets/c7342bb5-2b90-451f-92a4-287d924064e1)


# Applying K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

# Visualizing Clusters
plt.figure(figsize=(8, 8))
colors = ['green', 'red', 'yellow', 'violet', 'blue']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
for i in range(5):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=labels[i])

# Plotting Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
![Graph_file](https://github.com/user-attachments/assets/b382a5c1-f0d7-435b-bf4a-90ee75df5e45)

🎯 Key Insights

Customers are grouped into five different segments based on spending behavior and income.

Businesses can target high-income, high-spending customers for premium services.

Low-income, low-spending customers may need discount offers or promotional strategies.

Cluster Analysis helps in personalized marketing, better customer service, and revenue growth.

🚀 Future Enhancements

Feature Expansion: Include more features like Age, Gender, Purchase Frequency.

Real-Time Segmentation: Implement customer behavior tracking for dynamic segmentation.

📌 How to Use

Load the dataset into a Python environment.

Run the script to perform clustering and visualization.

Analyze the clusters and extract business insights.

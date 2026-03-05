import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")

# Select only required columns
X = data[['Annual Income (k$)','Spending Score (1-100)']]

# Remove rows that contain empty values
X = X.dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5)

data.loc[X.index, 'Cluster'] = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=data.loc[X.index,'Cluster'])

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()
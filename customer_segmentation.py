#step 1
'''import pandas as pd

data = pd.read_csv("customers.csv")
print(data.head())'''

#step 2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("customers.csv")

X = data[['Annual Income (k$)','Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5)

data['Cluster'] = kmeans.fit_predict(X)
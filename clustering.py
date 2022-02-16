from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

df = pd.read_csv('final2.csv')

x = df.iloc[:,[3,4]].values

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init="k-means++")
  kmeans.fit(x)
  
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), wcss, marker='o', color="red")
plt.title("The Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()
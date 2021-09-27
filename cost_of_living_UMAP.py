import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

sns.set(style='white', context='notebook', rc={'figure.figsize':(10,14)})

costs_of_living = pd.read_csv("cost-of-living.csv")

costs_of_living = costs_of_living.transpose()

cities = costs_of_living.index

pd.set_option("display.max_rows", None, "display.max_columns", None)

colors = []
for i in costs_of_living.Continent:
    if i == 1:
        colors.append('orange')
    if i == 2:
        colors.append('red')
    if i == 3:
        colors.append('blue')
    if i == 4:
        colors.append('green')
    if i == 5:
        colors.append('purple')
    if i == 6:
        colors.append('black')

asian = mpatches.Patch(color='red', label='Asian')
european = mpatches.Patch(color='blue', label='European')
south = mpatches.Patch(color='black', label='South American')
african = mpatches.Patch(color='orange', label='African')
north = mpatches.Patch(color='purple', label='North American')
australian = mpatches.Patch(color='green', label='Australian')

#costs_of_living.pop("Continent")



#print(costs_of_living["Saint Petersburg, Russia"][0]) dicts (keys are places) of dicts (keys are num of rows)
#print(costs_of_living)

import umap 

reducer = umap.UMAP()

costs_of_living_data = costs_of_living[costs_of_living.keys()].values


scaled_costs_of_living_data = StandardScaler().fit_transform(costs_of_living_data)

def draw_umap(n_neighbors=15, min_dist=0.1, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric
    )
    u = fit.fit_transform(scaled_costs_of_living_data )
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.scatter(u[:,0], u[:,1], c = colors)
    
    ax.legend(handles=[asian, african, north, south, european, australian])
    
    for i, txt in enumerate(cities):
        if (costs_of_living.Continent[i] == 2):
            ax.annotate(txt, (u[:,0][i], u[:,1][i]), size = 7)
    """
    fig3 = plt.figure(3)
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(scaled_costs_of_living_data)
    ax = fig3.add_subplot(111)
    ax.scatter(u[:,0], u[:,1],  c=kmeans.labels_, cmap='rainbow')


    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=3,
        metric=metric
    )
    fig2 = plt.figure(2)
    u = fit.fit_transform(scaled_costs_of_living_data )
    #colors=[sns.color_palette()[int(x)] for x in costs_of_living.Continent.map({1:0, 2:1, 3:2, 4:3, 5:4, 6:5})]
    
    ax = fig2.add_subplot(111, projection='3d')
    ax.scatter(u[:,0], u[:,1], u[:,2], 
    c=colors, s =100)
    ax.legend(handles=[asian, african, north, south, european, australian])

    #for i, txt in enumerate(cities):
    #    ax.annotate(txt, (u[:,0][i], u[:,1][i], u[:,2][i] ))
    
    plt.title(title, fontsize=18)
    """
    plt.show()
    
draw_umap(n_neighbors=100, min_dist=0.1, title='UMAP')

    
    
    #plt.scatter(u[:, 0], u[:, 1], u[:,2],
        #c=[sns.color_palette()[int(x)] for x in costs_of_living.Continent.map({1:0, 2:1, 3:2, 4:3, 5:4, 6:5})], 
        #)
    #plt.show()
    

"""
kmeans = KMeans(n_clusters=5)
kmeans.fit(u)

plt.scatter(u[:,0], u[:,1],  c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

plt.show()
"""

"""
for i, txt in enumerate(cities):
        if (costs_of_living.Continent[i] == 3):
            ax.annotate(txt, (u[:,0][i], u[:,1][i]), size = 7)
"""
"""
kmeans = KMeans(n_clusters=5)
kmeans.fit(embedding)

plt.scatter(embedding[:,0], embedding[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

plt.show()
"""
"""
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the costs of living dataset', fontsize=24)

plt.show()
"""



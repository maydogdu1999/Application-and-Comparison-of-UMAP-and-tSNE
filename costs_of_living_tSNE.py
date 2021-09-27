import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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


costs_of_living_data = costs_of_living[costs_of_living.keys()].values


scaled_costs_of_living_data = StandardScaler().fit_transform(costs_of_living_data)

embedded = TSNE(n_components=2, perplexity=5 ).fit_transform(scaled_costs_of_living_data)

fig = plt.figure("t-SNE")

ax = fig.add_subplot(111)
ax.scatter(embedded[:,0], embedded[:,1], c = colors)
ax.legend(handles=[asian, african, north, south, european, australian])

#for i, txt in enumerate(cities):
    #if (costs_of_living.Continent[i] == 2):
#   ax.annotate(txt, (embedded[:,0][i], embedded[:,1][i]), size = 7)

plt.title("t-SNE", fontsize=18)

plt.show()
import numpy as np
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt

def custom_visualize(framework):
  fig, ax = plt.subplots()
  ax.scatter(framework.coordinates[:,0], framework.coordinates[:,1], c='blue')
  
  for bond in framework.bonds:
    start, end = framework.coordinates[bond]
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
  
  # 固定された節点を赤色で表示
  for pin in framework.pins:
    ax.scatter(framework.coordinates[pin,0], framework.coordinates[pin,1], c='red', marker='D')
  
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Custom Visualization of the Framework')
  plt.show()

dim = 2
# 完全グラフの生成
G_comp = nx.complete_graph(5)
# position of sites
coordinates = 5*np.random.randn(dim*len(G_comp)).reshape(-1,2)
print("coordinates:", coordinates)
# list of sites between sites
bonds = np.array(list(G_comp.edges()))
print("bonds:", bonds)
# create a Framework object
F = rp.framework(coordinates, bonds)
# calculate the rigidity matrix
print ("rigidity matrix:\n",F.rigidityMatrix().T)
custom_visualize(F)
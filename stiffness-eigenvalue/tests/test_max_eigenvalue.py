import networkx as nx
import numpy as np
from stiffness_eigenvalue.max_eigenvalue import max_p_eigenvalue_lib

# 完全グラフでテスト
def test_complete():
  # np.random.seed(1)
  # 各定数
  d = 2
  V = 10
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # position of sites
  p = 5*np.random.randn(d*V).reshape(-1,d)
  # 円周上においてテスト 
  theta = np.linspace(0, 2 * np.pi, V+1)[:V]
  x_unit_circle = np.cos(theta).reshape(-1,1)
  y_unit_circle = np.sin(theta).reshape(-1,1)
  p_circle = np.hstack((x_unit_circle, y_unit_circle))
  p_circle[9,0] = 0.5
  p_circle[9,1] = 0.5 
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  p, eigen_val, eigen_vec = max_p_eigenvalue_lib(G_regular=G_comp, p=p, eigen_vec_0= v0,visual_eigen=True)
  
if __name__ == "__main__":
  test_complete()
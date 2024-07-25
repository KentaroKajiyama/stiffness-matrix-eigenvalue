import networkx as nx
import numpy as np
from stiffness_eigenvalue.max_eigenvalue

# 完全グラフでテスト
def test_complete():
  # np.random.seed(1)
  # 各定数
  eps = 1
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
  p, eigen_val, eigen_vec = max_p_eigenvalue_lib(G_regular=G_comp, p=p, eigen_vec_0= v0, eps=eps,visual_eigen=True)
  # テスト
  # p_circle, eigen_val, eigen_vec = max_p_eigenvalue_lib(G_regular=G_comp, p=p_circle, eigen_vec_0= v0, eps=eps,visual_eigen=True)
# k-random-regularグラフでテスト
def test_regular():
  # 各定数
  eps = 0.5
  d = 2
  k = 5
  n = 12
  # 完全グラフの生成
  G_regular = nx.random_regular_graph(k,n)
  # position of sites
  p = np.array([[220.26468935,142.08279461],
  [222.47367594, 133.31886717],
  [221.2007659,  146.26178173],
  [232.05277869, 127.71005911],
  [231.23487978, 145.55315542],
  [225.1127207,  139.77029442],
  [217.60549425, 147.38141886],
  [219.40312948, 136.5095388 ],
  [217.61356694, 140.03527164],
  [210.29054027, 128.6630659 ],
  [228.35735623, 131.02726757],
  [227.59364421, 147.54974161]])
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*n)
  p, eigen_val, eigen_vec = max_p_eigenvalue(G_regular=G_regular, p=p, eigen_vec_0= v0, eps=eps,visual_eigen=True)
  print("eigen_val:", eigen_val)
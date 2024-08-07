from old.old_min_non_zero_eigenvalue import non_zero_eigenvalue, gradient_descent
from stiffness_eigenvalue.visualize import custom_visualize
import numpy as np
import networkx as nx
import rigidpy as rp
# 簡単な半正定値行列でテスト（固有値があっているかも確認する固有値は8と1なので1が出力されてほしい）
def test_1():
  L = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
  v0 = np.array([9,1,1])/3
  print("L:",L)
  eigen_val_lib, eigen_vec_lib = non_zero_eigenvalue(L, v0)
  eigen_val_hand, eigen_vec_hand = gradient_descent(v0, L)
  print("eigen_val_lib:", eigen_val_lib)
  print("eigen_val_hand:", eigen_val_hand)
  print("eigen_vec_lib:", eigen_vec_lib)
  print("eigen_vec_hand:", eigen_vec_hand)

# 完全グラフでテスト
def test_2(dim, size_n):
  # 各定数
  eps = 1
  d = dim
  V = size_n
  # 完全グラフの生成
  G_comp = nx.complete_graph(5)
  # position of sites
  coordinates = 5*np.random.randn(d*V).reshape(-1,d)
  # list of sites between sites
  bonds = np.array(list(G_comp.edges()))
  # create a Framework object
  F = rp.framework(coordinates, bonds)
  # calculate the rigidity matrix
  R = F.rigidityMatrix().T
  L = R @ R.T
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  stiff_approx_matrix = L+eps*np.eye(d*V)
  eigen_val_lib, eigen_vec_lib = non_zero_eigenvalue(stiff_approx_matrix, v0)

# k-random-regularグラフでテスト
def test_3():
  # 各定数
  eps = 1
  d = 2
  k = 3
  V = 8
  # k-regular graphの生成
  G_regular = nx.random_regular_graph(k,V)
  # position of sites
  coordinates = 5*np.random.randn(d*V).reshape(-1,d)
  print("coordinates:", coordinates)
  # list of sites between sites
  bonds = np.array(list(G_regular.edges()))
  print("bonds:", bonds)
  # create a Framework object
  F = rp.framework(coordinates, bonds)
  # calculate the rigidity matrix
  R = F.rigidityMatrix().T
  L = R @ R.T
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  print("v0:",v0)
  stiff_approx_matrix = L+eps*np.eye(d*V)
  eigen_val_lib, eigen_vec_lib = non_zero_eigenvalue(stiff_approx_matrix, v0)
  eigen_val_hand, eigen_vec_hand = gradient_descent(v0, stiff_approx_matrix)
  print("eigen_val_lib:", eigen_val_lib)
  print("eigen_val_hand:", eigen_val_hand)
  print("eigen_vec_lib:", eigen_vec_lib)
  print("eigen_vec_hand:", eigen_vec_hand)
  custom_visualize(F)


# 時間の計測
# テスト実行用
if __name__ == "__main__":
  test_1()

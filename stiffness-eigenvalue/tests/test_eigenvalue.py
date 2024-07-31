import numpy as np
import scipy
import networkx as nx
import math
from stiffness_eigenvalue.eigenvalue import gen_skew_symmetric, gen_parallel_vector, gen_basis, min_non_zero_eigen
from stiffness_eigenvalue.framework import stiffness_matrix

# Sとtの生成のテスト
def test1():
  dim = 2
  S_box = gen_skew_symmetric(dim)
  t_box = gen_parallel_vector(dim)
  for i, trans in enumerate(zip(S_box,t_box)):
    S = trans[0]; t = trans[1]
    print(f"{i}th transformer: S = {S}, t = {t}")
# 基底生成のテスト
def test2():
  dim = 4
  V = 10
  # position of sites
  p = 5*np.random.randn(dim*V).reshape(-1,dim)
  basis_box = gen_basis(dim,p)
  for i, basis in enumerate(basis_box):
    print(f"{i}th basis: {basis}")
# 最小固有値のテスト
def test3():
  # 各定数
  d = 2
  V = 6
  D = math.comb(d+1,2)
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # 辺集合
  bonds = np.array(list(G_comp.edges()))
  # position of sites
  p = 5*np.random.randn(d*V).reshape(-1,d)
  # 初期固有ベクトル
  x0 = 3*np.random.randn(d*V)
  # stiffness matrix
  L = stiffness_matrix(p, bonds)
  # ライブラリーと手動実装の比較
  # ライブラリー
  lib_eigvals, lib_eigvecs = scipy.linalg.eigh(L)
  sorted_indices = np.argsort(lib_eigvals)
  lib_eigvals_sorted, lib_eigvecs_sorted = lib_eigvals[sorted_indices], lib_eigvecs[sorted_indices]
  lib_eigval, lib_eigvec = lib_eigvals_sorted[D], lib_eigvecs_sorted[D] # 0 baseなのでD+1番目だがインデックスはDであることに注意
  # 手動実装
  hand_eigval, hand_eigvec = min_non_zero_eigen(L, x0, d, p)
  print("library vs hand")
  print("Eigenvalue")
  print(f"lib:{lib_eigval}, hand:{hand_eigval}")
  print("Eigenvector")
  print(f"lib:{lib_eigvec}, hand:{hand_eigvec}")
  

if __name__ == "__main__":
  test3()
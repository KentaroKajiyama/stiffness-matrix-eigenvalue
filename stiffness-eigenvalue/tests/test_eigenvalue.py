import numpy as np
import scipy
import networkx as nx
import math
from stiffness_eigenvalue.eigenvalue import gen_skew_symmetric, gen_parallel_vector, gen_basis, min_non_zero_eigen
from stiffness_eigenvalue.framework import stiffness_matrix
import time

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
  d = 2
  V = 3
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # 辺集合
  bonds = np.array(list(G_comp.edges()))
  # position of sites
  p = 5*np.random.randn(d*V).reshape(-1,d)
  # stiffness matrix
  L = stiffness_matrix(p, bonds)
  # Eigenvalues
  eigen_vals, eigen_vecs = np.linalg.eig(L)
  # eigen_vals[i] is the eigenvalue corresponding to the eigenvector eigen_vecs[:,i]
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices = np.argsort(eigen_vals)
  eigen_vals, eigen_vecs = eigen_vals[sorted_indices], eigen_vecs[:,sorted_indices]
  basis_box = gen_basis(d,p)
  for i, basis in enumerate(basis_box):
    print(f"{i+1}th basis: {basis}")
  # # 直交性の確認（kernelの基底と正固有値の固有ベクトルが直交することの確認）
  print(f"L,1:{np.dot(L, basis_box[0])}")
  print(f"L,2:{np.dot(L, basis_box[1])}")
  print(f"L,3:{np.dot(L, basis_box[2])}")
  print(f"L,4:{np.dot(L, eigen_vecs[:,3])}, eigenvalue:{eigen_vals[3]}, eigenvector:{eigen_vecs[:,3]}, lambda*v:{eigen_vals[3]*eigen_vecs[:,3]}")
  print(f"1,4:{np.dot(basis_box[0], eigen_vecs[:,3])}")
  print(f"2,4:{np.dot(basis_box[1], eigen_vecs[:,3])}")
  print(f"3,4:{np.dot(basis_box[2], eigen_vecs[:,3])}")
  
  
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
  start = time.time()
  lib_eigvals, lib_eigvecs = scipy.linalg.eigh(L)
  sorted_indices = np.argsort(lib_eigvals)
  lib_eigvals_sorted, lib_eigvecs_sorted = lib_eigvals[sorted_indices], lib_eigvecs[:, sorted_indices]
  lib_eigval, lib_eigvec = lib_eigvals_sorted[D], lib_eigvecs_sorted[:,D] # 0 baseなのでD+1番目だがインデックスはDであることに注意
  end1 = time.time()
  # 手動実装
  hand_eigval, hand_eigvec = min_non_zero_eigen(L, x0, d, p)
  end2 = time.time()
  print("library vs hand")
  print("Eigenvalue")
  print(f"lib:{lib_eigval}, hand:{hand_eigval}")
  print("Eigenvector")
  print(f"lib:{lib_eigvec}, hand:{hand_eigvec}")
  print("vLv")
  print(f"lib:{np.dot(lib_eigvec, np.dot(L, lib_eigvec))}, hand:{np.dot(hand_eigvec, np.dot(L, hand_eigvec))}")
  print(f"norm lib:{np.linalg.norm(lib_eigvec)}, norm hand:{np.linalg.norm(hand_eigvec)}")
  print(f"lib time:{end1-start}, hand time:{end2-end1}")

if __name__ == "__main__":
  test3()
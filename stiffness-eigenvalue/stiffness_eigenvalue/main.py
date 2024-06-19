import numpy as np
import networkx as nx
import matplotlib
from max_eigenvalue import max_p_eigenvalue
from max_eigenvalue import plot_eigen_vals

# k-regular-graphで計算
def main():
  # 各定数
  eps = 1.0
  ITER_EPS = 20
  k = 5
  d = 2
  MAXITER = 100
  # 固有値の格納
  eigen_vals = []
  for n in range(12,MAXITER):
    try:
      # k-regularグラフの生成
      G_regular = nx.random_regular_graph(k, n)
    except nx.NetworkXError:
      print(f"Skipping n={n} as it does not satisfy the condition for a {k}-regular graph.")
      continue
    # position of sites
    p = 5*np.random.randn(d*n).reshape(-1,d)
    # 初期固有ベクトル
    eigen_vec = 3*np.random.randn(d*n)
    eigen_val = 0
    for i in range(ITER_EPS):
      p, eigen_val, eigen_vec = max_p_eigenvalue(G_regular=G_regular, p=p, eigen_vec_0= eigen_vec, eps=eps)
      eps /= 2
    eigen_vals.append(eigen_val)
  plot_eigen_vals(eigen_vals)
  
  # k-complete-graphで計算
def test():
  # 各定数
  eps = 1.0
  ITER_EPS = 20
  k = 5
  d = 2
  MAXITER = 10**6
  # 固有値の格納
  eigen_vals = []
  for n in range(12,MAXITER,100):
    try:
      # completeグラフの生成
      G_regular = nx.complete_graph(n)
    except nx.NetworkXError:
      print(f"Skipping n={n} as it does not satisfy the condition for a {k}-regular graph.")
      continue
    # position of sites
    p = 5*np.random.randn(d*n).reshape(-1,d)
    # 初期固有ベクトル
    eigen_vec = 3*np.random.randn(d*n)
    eigen_val = 0
    for i in range(ITER_EPS):
      p, eigen_val, eigen_vec = max_p_eigenvalue(G_regular=G_regular, p=p, eigen_vec_0= eigen_vec, eps=eps)
      eps /= 2
    eigen_vals.append(eigen_val)
  plot_eigen_vals(eigen_vals)
  
if __name__ == "__main__":
  test()

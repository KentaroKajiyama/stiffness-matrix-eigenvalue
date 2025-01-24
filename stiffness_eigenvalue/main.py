import numpy as np
import networkx as nx
from stiffness_eigenvalue.max_eigenvalue_lib import max_p_eigenvalue_lib
from stiffness_eigenvalue.visualize import plot_eigen_vals_and_alpha
# k-regular-graphで計算
def main():
  # 各定数
  k = 8
  d = 2
  max_iters_for_armijo = [8, 16, 25]
  # 
  for max_iter_for_armijo in max_iters_for_armijo:
    is_created = False
    for n in range(10, 2000):
      try:
        if is_created:
          break
      # k-regularグラフの生成
        G_regular = nx.random_regular_graph(k, n)
        # position of sites
        p = 5*np.random.randn(d*n).reshape(-1,d)
        # 固有値計算
        p_optimized, max_eigenvalue, eigen_val_box, alpha_box, multiplicity_box = max_p_eigenvalue_lib(G_regular=G_regular, p=p, max_iter_for_armijo=max_iter_for_armijo)
        # 画像の prefix 
        save_prefix = f"d={d}_k={k}_n={n}_alpha-iter={max_iter_for_armijo}"
        plot_eigen_vals_and_alpha(eigen_val_box, alpha_box, multiplicity_box, is_save=True, save_prefix=save_prefix)
        is_created = True
        print(f"n = {n}")
      except nx.NetworkXError as e:
        print(f"Skipping n={n} as it does not satisfy the condition for a {k}-regular graph.")
        print(e)
        continue
    
  
if __name__ == "__main__":
  main()

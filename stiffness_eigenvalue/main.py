import numpy as np
import networkx as nx
from max_eigenvalue_lib_with_timeout_windows import max_p_eigenvalue_lib
from visualize import plot_eigen_vals_and_alpha
import sys

class Tee:
  def __init__(self, filename):
    self.file = open(filename, "w")
    self.stdout = sys.stdout

  def write(self, data):
    self.stdout.write(data)  # コンソールに出力
    self.file.write(data)  # ファイルに出力

  def flush(self):
    self.stdout.flush()
    self.file.flush()
# k-regular-graphで計算
def main():
  sys.stdout = Tee("output.log")
  # 各定数
  k = 8
  d = 2
  max_iters_for_armijo = [8, 16, 25]
  # 
  for max_iter_for_armijo in max_iters_for_armijo:
    is_created = False
    for n in range(1000, 1100):
      try:
        if is_created:
          break
        print(f"Experiment starts k={k}, d={d}, n={n} G is k-random-regular")
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
  sys.stdout = sys.stdout.stdout
  
if __name__ == "__main__":
  main()

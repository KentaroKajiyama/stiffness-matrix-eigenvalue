import networkx as nx
import numpy as np
from stiffness_eigenvalue.max_eigenvalue_lib_clean import max_p_eigenvalue_lib
from old.max_eigenvalue_hand import max_p_eigenvalue_hand
from old.max_eigenvalue_lib_legacy import max_p_eigenvalue_lib_legacy
from stiffness_eigenvalue.visualize import plot_eigen_vals_and_alpha, custom_visualize
import rigidpy as rp
from matplotlib import pyplot as plt

# 完全グラフでテスト（手動で実装したもの）
def test_complete_hand():
  # np.random.seed(1)
  # 各定数
  d = 2
  V = 10
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # position of sites
  p = 5*np.random.randn
  (d*V).reshape(-1,d)
  # 円周上においてテスト 
  theta = np.linspace(0, 2 * np.pi, V+1)[:V]
  x_unit_circle = np.cos(theta).reshape(-1,1)
  y_unit_circle = np.sin(theta).reshape(-1,1)
  p_circle = np.hstack((x_unit_circle, y_unit_circle))
  p_circle[9,0] = 0.5
  p_circle[9,1] = 0.5 
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  p, eigen_val, eigen_vec = max_p_eigenvalue_hand(G_regular=G_comp, p=p, eigen_vec_0=v0, visual_eigen=True)
  
# 完全グラフでテスト（ライブラリを用いたもの）（time: 200s~300s）=> Backtrackの更新回数を1回に、libraryを対称行列限定のものにするとと150s~170s程度に改善全体の1/4~1/2程度の時間をArmijo条件の反復に費やしていそう。
def test_complete_lib():
  # np.random.seed(1)
  # 各定数
  d = 2
  V = 10
  p = []; max_eigen = 0; eigen_val_box = []; alpha_box = []; multiplicity_box = [];eigen_record_box = []
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # 辺集合
  bonds = np.array(list(G_comp.edges()))
  for i in range(20):
    # position of sites
    max_previous = max_eigen
    p_init = 2*np.random.randn(d*V).reshape(-1,d)
    p_current, max_eigen_current, eigen_val_box_current, alpha_box_current, multiplicity_box_current = max_p_eigenvalue_lib(G_regular=G_comp, p=p_init,visual_eigen=False)
    eigen_record_box.append(max_eigen_current)
    if i==0 or (i > 0 and max_eigen_current > max_previous) :
      p = p_current
      max_eigen = max_eigen_current
      eigen_val_box = eigen_val_box_current
      alpha_box = alpha_box_current
      multiplicity_box = multiplicity_box_current
  plt.plot(eigen_record_box)
  plot_eigen_vals_and_alpha(eigen_val_box, alpha_box, multiplicity_box)
  F = rp.framework(p, bonds)
  custom_visualize(F, label=f"opt, index value")
  

  
  # 完全グラフでテスト（ライブラリを用いたもの）（time: 200s~300s）=> Backtrackの更新回数を1回に、libraryを対称行列限定のものにするとと150s~170s程度に改善全体の1/4~1/2程度の時間をArmijo条件の反復に費やしていそう。
def test_complete_lib_legacy():
  # np.random.seed(1)
  # 各定数
  d = 2
  V = 14
  # 完全グラフの生成
  G_comp = nx.complete_graph(V)
  # position of sites
  p = 5*np.random.randn(d*V).reshape(-1,d)
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  p, eigen_val, eigen_vec = max_p_eigenvalue_lib_legacy(G_regular=G_comp, p=p, eigen_vec_0= v0,visual_eigen=True)
  
if __name__ == "__main__":
  test_complete_lib()
  
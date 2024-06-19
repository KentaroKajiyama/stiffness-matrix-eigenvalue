import numpy as np
from min_non_zero_eigenvalue import non_zero_eigenvalue, gradient_descent
import rigidpy as rp
import matplotlib.pyplot as plt
import time
import networkx as nx
from min_non_zero_eigenvalue import custom_visualize

# G:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue(G_regular, p, eigen_vec_0, eps, visual_eigen=False, visual_frame=False):
  # 各定数
  # 最大繰り返し回数
  MAX_ITER = 100
  # 移動分の係数
  alpha = 0.2
  # realization,固有値の格納（格納する固有値は初回を含めて100回分）
  p_box = []
  eigen_vals = []
  eigen_vecs = []
  # 次元 dim
  dim = p.shape[1]
  # node数
  V = len(G_regular)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # 固有ベクトルの初期化
  eigen_vec = eigen_vec_0
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    # framework
    F = rp.framework(p, bonds)
    if visual_frame:
      custom_visualize(F)
    # Rigidity matrix
    R = F.rigidityMatrix().T
    # Stiffness matrix
    L = R @ R.T
    # Approximate stiffness matrix
    L_approx = L + eps*np.eye(dim*V)
    eigen_val, eigen_vec = non_zero_eigenvalue(L_approx, eigen_vec)
    # realizationの更新
    p -= alpha*np.copy(eigen_vec).reshape(-1,dim)
    # 固有値の格納
    eigen_vals.append(eigen_val)
    eigen_vecs.append(eigen_vec)
    # フレームワークの削除
    del F
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_vals)
  # visual = Trueの場合に固有値の推移の様子をプロットする。
  if visual_eigen:
    plot_eigen_vals(eigen_vals)
  return p_box[max_index], eigen_vals[max_index], eigen_vecs[max_index]
# ライブラリーを用いてテスト
# G:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue_lib(G_regular, p, eigen_vec_0, eps, visual_eigen=False, visual_frame=False):
  # 各定数
  # 最大繰り返し回数
  MAX_ITER = 1000
  # 移動分の係数
  alpha = 0.2
  # realization,固有値の格納（格納する固有値は初回を含めて100回分）
  p_box = []
  eigen_val_box = []
  eigen_vec_box = []
  # 次元 dim
  dim = p.shape[1]
  print("dim:",dim)
  # node数
  V = len(G_regular)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # 固有ベクトルの初期化
  eigen_vec = eigen_vec_0
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    # framework
    F = rp.framework(p, bonds)
    if visual_frame:
      custom_visualize(F)
    # Rigidity matrix
    R = F.rigidityMatrix().T
    # Stiffness matrix
    L = R @ R.T
    # Eigenvalues
    eigen_vals, eigen_vecs = np.linalg.eig(L)
    # Sort eigenvalues and eigenvectors in the decending order.
    sorted_indices = np.argsort(eigen_vals)
    eigen_vals, eigen_vecs = eigen_vals[sorted_indices], eigen_vecs[sorted_indices]
    # d=2 4th minimum eigenvalue
    fourth_smallest_eigenvalue = eigen_vals[3]
    fourth_smallest_eigenvector = eigen_vecs[:, 3].real
    print("fourth_smallest_eigenvalue:", fourth_smallest_eigenvalue)
    print("fourth_smallest_eigenvector:", fourth_smallest_eigenvector)
    # realizationの更新
    p -= alpha*np.copy(fourth_smallest_eigenvector).reshape(-1,dim)
    # 固有値の格納
    eigen_val_box.append(fourth_smallest_eigenvalue)
    eigen_vec_box.append(fourth_smallest_eigenvector)
    # フレームワークの削除
    del F
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  # visual = Trueの場合に固有値の推移の様子をプロットする。
  if visual_eigen:
    plot_eigen_vals(eigen_val_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F)
  return p_box[max_index], eigen_val_box[max_index], eigen_vec_box[max_index]
# 固有値のプロット
def plot_eigen_vals(eigen_vals, limit=False):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(1,1,1)
  ax1.plot(eigen_vals, label="eigenvalues")
  ax1.set_xlabel("index")
  ax1.set_ylabel("eigenvalue")
  ax1.legend(loc="best")
  fig.canvas.mpl_connect("key_press_event", on_key)
  if limit:
    plt.show(block=False)
    time.sleep(10)
    plt.close(fig)
  else:
    plt.show()

# Enterを押した際にfigureを消去
def on_key(event):
  if event.key == "enter":
    plt.close(event.canvas.figure)

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
  # 初期固有ベクトル
  v0 = 3*np.random.randn(d*V)
  p, eigen_val, eigen_vec = max_p_eigenvalue_lib(G_regular=G_comp, p=p, eigen_vec_0= v0, eps=eps,visual_eigen=True)

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

if __name__=="__main__":
  test_complete()
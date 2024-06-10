import numpy as np
from min_non_zero_eigenvalue import non_zero_eigenvalue
import rigidpy as rp
import matplotlib.pyplot as plt
import time

# G:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue(G_regular, p, eps, visual=False):
  # 各定数
  # 最大繰り返し回数
  MAX_ITER = 100
  # 移動分の係数
  alpha = 0.2
  # realization,固有値の格納（格納する固有値は初回を含めて100回分）
  p_box = []
  eigen_vals = []
  # 次元 dim
  dim = p.shape[1]
  # node数
  V = len(G_regular)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    # framework
    F = rp.framework(p, bonds)
    # Rigidity matrix
    R = F.rigidityMatrix().T
    # Stiffness matrix
    L = R @ R.T
    # Approximate stiffness matrix
    L_approx = L + eps*np.eye(dim*V)
    eigen_val, eigen_vec = non_zero_eigenvalue(L_approx,p)
    # realizationの更新
    p += alpha*eigen_vec
    # 固有値の格納
    eigen_vals.append(eigen_val)
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_vals)
  # visual = Trueの場合に固有値の推移の様子をプロットする。
  if visual:
    plot_eigen_vals(eigen_vals)
  return p_box[max_index], eigen_vals[max_index]

# 固有値のプロット
def plot_eigen_vals(eigen_vals, limit=False):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(1,1,1)
  ax1.plot(y=eigen_vals, label="eigenvalues")
  ax1.set_xlabel("index")
  ax1.set_ylabel("eigenvalue")
  ax1.legend(loc="best")
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show(block=False)
  if limit:
    time.sleep(10)
    plt.close(fig)

# Enterを押した際にfigureを消去
def on_key(event):
  if event.key == "enter":
    plt.close(event.canvas.figure)


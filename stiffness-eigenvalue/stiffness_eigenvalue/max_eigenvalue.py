import numpy as np
from min_non_zero_eigenvalue import non_zero_eigenvalue, gradient_descent
import rigidpy as rp
import matplotlib.pyplot as plt
import time
import networkx as nx
from min_non_zero_eigenvalue import custom_visualize

def pseudo_descent_dir(p, bonds, eigenvector, dim, epsilon=0.0000001):
  # +の場合
  p_plus = p - epsilon*eigenvector.reshape(-1, dim)
  # フレームワーク・剛性行列の作成
  F_plus = rp.framework(p_plus, bonds)
  # Rigidity matrix
  R_plus = F_plus.rigidityMatrix().T
  # Stiffness matrix
  L_plus = R_plus @ R_plus.T
  # Eigenvalues
  eigen_vals_plus, eigen_vecs_plus = np.linalg.eig(L_plus)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_plus = np.argsort(eigen_vals_plus)
  eigen_vals_plus = eigen_vals_plus[sorted_indices_plus]
  # -の場合
  p_minus = p - epsilon*eigenvector.reshape(-1, dim)
  # フレームワーク・剛性行列の作成
  F_minus = rp.framework(p_minus, bonds)
  # Rigidity matrix
  R_minus = F_minus.rigidityMatrix().T
  # Stiffness matrix
  L_minus = R_minus @ R_minus.T
  # Eigenvalues
  eigen_vals_minus, eigen_vecs_minus = np.linalg.eig(L_minus)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_minus = np.argsort(eigen_vals_minus)
  eigen_vals_minus = eigen_vals_minus[sorted_indices_minus]
  if eigen_vals_plus[3] >= eigen_vals_minus[3]:
    return 1
  else:
    return -1
  
def pseudo_armijo(alpha, dim, p, bonds):
  # 最大繰り返し回数
  MAX_DIVIDE = 20; time =0; row = 0.6;
  # Armijoの係数定数c1
  c1 = 0.1
  # 2方向を考慮
  alpha_plus = alpha
  alpha_minus = -alpha
  # フレームワーク・剛性行列の作成
  F = rp.framework(p, bonds)
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
  # eigenvalueの方向決め
  dd = pseudo_descent_dir(p, bonds, fourth_smallest_eigenvector, dim)
  fourth_smallest_eigenvector *= dd
  # 更新後のrealization p_after
  p_after = p-alpha*fourth_smallest_eigenvector.reshape(-1, dim)
  # 更新後のフレームワークを用意
  F_after = rp.framework(p_after, bonds)
  # Rigidity matrix
  R_after = F_after.rigidityMatrix().T
  # Stiffness matrix
  L_after = R_after @ R_after.T
  # Eigenvalues
  eigen_vals_after, eigen_vecs_after = np.linalg.eig(L_after)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_after = np.argsort(eigen_vals_after)
  eigen_vals_after, eigen_vecs_after = eigen_vals_after[sorted_indices_after], eigen_vecs_after[sorted_indices_after]
  # d=2 4th minimum eigenvalue
  fourth_smallest_eigenvalue_after, fourth_smallest_eigenvector_after = eigen_vals_after[3], eigen_vecs_after[:,3]
  cd = fourth_smallest_eigenvalue + c1*alpha*np.dot(fourth_smallest_eigenvector, p.flatten()) - fourth_smallest_eigenvalue_after
  while(cd>0 and time < MAX_DIVIDE):
    alpha *= row
    p_after = p-alpha*fourth_smallest_eigenvector.reshape(-1,dim)
    F_after = rp.framework(p_after, bonds)
    # Rigidity matrix
    R_after = F_after.rigidityMatrix().T
    # Stiffness matrix
    L_after = R_after @ R_after.T
    # Eigenvalues
    eigen_vals_after, eigen_vecs_after = np.linalg.eig(L_after)
    # Sort eigenvalues and eigenvectors in the decending order.
    sorted_indices_after = np.argsort(eigen_vals_after)
    eigen_vals_after, eigen_vecs_after = eigen_vals_after[sorted_indices_after], eigen_vecs_after[sorted_indices_after]
    # d=2 4th minimum eigenvalue
    fourth_smallest_eigenvalue_after = eigen_vals_after[3]
    fourth_smallest_eigenvector_after = eigen_vecs_after[:, 3].real
    cd = fourth_smallest_eigenvalue + c1*alpha*np.dot(fourth_smallest_eigenvector, p.flatten()) - fourth_smallest_eigenvalue_after
    time +=1
  del F
  del F_after
  return alpha, fourth_smallest_eigenvalue_after, fourth_smallest_eigenvector_after, p_after

# ライブラリーを用いてテスト
# G:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue_lib(G_regular, p, eigen_vec_0, eps, visual_eigen=False, visual_frame=False):
  # 各定数
  # 最初のp
  p_init = p
  # 最大繰り返し回数
  MAX_ITER = 3000
  # realization,固有値の格納（格納する固有値は初回を含めて100回分）
  p_box = []
  alpha_box = []
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
  # pの正規化
  p = p/np.linalg.norm(p)
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    # 移動分の係数
    alpha = 1
    alpha, fourth_smallest_eigenvalue_after, fourth_smallest_eigenvector_after, p_after = pseudo_armijo(alpha, dim, p, bonds)
    alpha_box.append(alpha)
    # realizationを正規化して更新
    p = p_after/np.linalg.norm(p_after)
    # 固有値の格納
    eigen_val_box.append(fourth_smallest_eigenvalue_after)
    eigen_vec_box.append(fourth_smallest_eigenvector_after)
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  # visual = Trueの場合に固有値の推移の様子をプロットする。
  if visual_eigen:
    plot_eigen_vals(eigen_val_box)
    plot_alpha(alpha_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F, label="opt")
    F = rp.framework(p_init, bonds)
    custom_visualize(F, label = "init")
  return p_box[max_index], eigen_val_box[max_index], eigen_vec_box[max_index]
# G:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue(G_regular, p, eigen_vec_0, eps, visual_eigen=False, visual_frame=False):
  # 各定数
  # 最大繰り返し回数
  MAX_ITER = 100
  # 移動分の係数
  alpha = 1
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
    alpha = pseudo_armijo(alpha, dim, p, bonds)
    print("alpha:",alpha)
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
# alphaのプロット
def plot_alpha(alpha_box, limit=False):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(1,1,1)
  ax1.plot(alpha_box, label="alpha")
  ax1.set_xlabel("index")
  ax1.set_ylabel("alpha")
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

if __name__=="__main__":
  test_complete()
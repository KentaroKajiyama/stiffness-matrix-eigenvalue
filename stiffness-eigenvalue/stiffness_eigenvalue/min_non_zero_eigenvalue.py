import numpy as np
from scipy.optimize import minimize
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
import rigidpy as rp
import math
import time

# 極小正数
eps = 10**(-15)
# 目的関数（最小化問題にするために係数に-をつけている）
def objective(v,L):
  if np.linalg.norm(v) <= eps:
    return 0
  else:
    return -np.dot(v.T, v) / np.dot(v.T, np.dot(L, v)) 
# ライブラリ
def non_zero_eigenvalue(L, v0):
  # 目的関数
  obj_func = partial(objective, L=L)
  # 最適化の実行（目的関数のマイナスの値を最小化問題に）
  solution = minimize(obj_func, v0)
  # 結果の表示（固有ベクトルはおｋ、結果は負の逆数を取って元の最小非ゼロ固有値を求めるようにしている。）
  return 1/(-solution.fun), solution.x

# 目的関数の勾配（係数に-をつけた目的関数の勾配)
def objective_grad(v, L):
  if np.linalg.norm(v) <= eps:
    return np.zeros_like(v)
  else:
    numerator = -2 * v / np.dot(v.T, np.dot(L, v))
    denominator = 2 * np.dot(v.T, v) * np.dot(L, v) / (np.dot(v.T, np.dot(L, v)) ** 2)
    return numerator + denominator

# ライブラリが上手く動かないので手動で再急降下法を実装
# Armijo条件を使用して学習率を調整
def armijo_line_search(v, L, grad, alpha=1.0, beta=0.8, sigma=1e-4):
  while objective(v - alpha * grad, L) > objective(v, L) - sigma * alpha * np.dot(grad.T, grad):
    alpha *= beta
  return alpha

# 再急降下法
def gradient_descent(v_init, L, max_iter=1000, tol=1e-6):
  v = v_init
  print("v_init:", v_init)
  print("L:",L)
  for i in range(max_iter):
    grad = objective_grad(v, L)
    alpha = armijo_line_search(v, L, grad)  # Armijo条件に基づく学習率の決定
    v_new = v - alpha * grad
    # 収束判定
    if np.linalg.norm(v_new - v) < tol:
      print(f"Converged at iteration {i}")
      break
    v = v_new
  print("v:",v)
  return -1/objective(v,L), v

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
def test_2():
  # 各定数
  eps = 1
  d = 2
  V = 5
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

# フレームワークの可視化用
def custom_visualize(framework, limit=False):
  fig, ax = plt.subplots()
  ax.scatter(framework.coordinates[:,0], framework.coordinates[:,1], c='blue')
  
  for bond in framework.bonds:
    start, end = framework.coordinates[bond]
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
  
  # 固定された節点を赤色で表示
  for pin in framework.pins:
    ax.scatter(framework.coordinates[pin,0], framework.coordinates[pin,1], c='red', marker='D')
  
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Custom Visualization of the Framework')
  fig.canvas.mpl_connect("key_press_event", on_key)
  plt.show(block=False)
  if limit:
    plt.show(block=False)
    time.sleep(10)
    plt.close(fig)
  else:
    plt.show()

def on_key(event):
  if event.key == 'enter':
    plt.close(event.canvas.figure)

# regularグラフでテスト
if __name__ == "__main__":
  test_3()

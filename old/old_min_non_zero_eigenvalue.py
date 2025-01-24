import numpy as np
from scipy.optimize import minimize
from functools import partial

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


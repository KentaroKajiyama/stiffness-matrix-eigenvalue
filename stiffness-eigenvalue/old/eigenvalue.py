import numpy as np
import math
from scipy.optimize import minimize
from functools import partial
from joblib import Parallel, delayed
import time

"""
EPS_OBJ : A small constant for objective function 
BIG_C : A big constant for optimization
"""
###########################
EPS_OBJ = 10**(-8)
BIG_C = 100
###########################

# 回転行列（歪対称行列）の生成
def gen_skew_symmetric(d):
  result = []
  for i in range(0, d):
    for j in range(i+1,d):
      S = np.zeros((d,d))
      S[i][j] = 1; S[j][i] = -1;
      result.append(S)
  # 平行移動用の余分なS（ただのゼロ行列）
  for i in range(0,d):
    result.append(np.zeros(shape=(d,d)))
  return result
# 平行移動ベクトルの生成
def gen_parallel_vector(d):
  # 回転用の余分なt（ただの0ベクトル）
  if d > 1:
    result = [np.zeros(d) for _ in range(0,math.comb(d,2))]
  else:
    result = []
  [result.append(np.array([0 if i!=j else 1 for i in range(0,d)])) for j in range(0,d)]
  return result
# 基底の生成（Sとtはグローバルに使いまわす。）
def process_single_basis(S, t, p):
  x = np.dot(S, p.T).T + t
  x_flat = x.flatten()
  norm = np.linalg.norm(x_flat)
  if norm > 0:
    x_flat /= norm
  return x_flat
def gen_basis(S_box, t_box, p):
  n_jobs = -1
  basis_box = Parallel(n_jobs=n_jobs)(delayed(process_single_basis)(S, t, p) for S, t in zip(S_box, t_box))
  return basis_box

# 目的関数
def objective(x, L):
  norm_x = np.linalg.norm(x)
  if norm_x <= EPS_OBJ:
    return 0
  x.flatten()
  Lx = np.dot(L, x)
  return np.dot(x.T, Lx) / (norm_x**2)
def objective_grad(x, L):
  norm_x = np.linalg.norm(x)
  if norm_x <= EPS_OBJ:
    return np.zeros_like(x)
  Lx = np.dot(L, x)
  grad = 2 * (Lx / norm_x**2 - np.dot(x.T, Lx) * x / norm_x**4)
  return grad
def min_non_zero_eigen(L, x0, p, S_box, t_box):
  # 近似行列の生成
  basis_box = gen_basis(S_box, t_box, p)
  L_tilde = L
  for basis in basis_box:
    L_tilde += BIG_C * (basis[:, np.newaxis] * basis[np.newaxis, :])
  # 目的関数
  start = time.time()
  obj_func = partial(objective, L=L_tilde)
  opt = minimize(obj_func, x0, method="Newton-CG", jac=partial(objective_grad, L=L_tilde), options={"disp": False})
  eigenvector = opt.x/np.linalg.norm(opt.x)
  end = time.time()
  print(f"Elapsed time for eigenvalue calculation:{end-start}")
  return opt.fun, eigenvector

def objective_sparce(x, L):
  norm_x = np.linalg.norm(x)
  if norm_x <= EPS_OBJ:
    return 0
  x.flatten()
  Lx = L.dot(x)
  return np.dot(x.T, Lx) / (norm_x**2)

def objective_grad_sparce(x, L):
  norm_x = np.linalg.norm(x)
  if norm_x <= EPS_OBJ:
    return np.zeros_like(x)
  Lx = L.dot(x)
  grad = 2 * (Lx / norm_x**2 - np.dot(x.T, Lx) * x / norm_x**4)
  return grad
# 固有値計算（最適化による計算）。固有値と固有ベクトルを返す。
def min_non_zero_eigen_sparce(L, x0, p, S_box, t_box):
  # 近似行列の生成
  basis_box = gen_basis(S_box, t_box, p)
  L_tilde = L
  for basis in basis_box:
    L_tilde += BIG_C * (basis[:, np.newaxis] * basis[np.newaxis, :])
  # 目的関数
  start = time.time()
  obj_func = partial(objective_sparce, L=L_tilde)
  opt = minimize(obj_func, x0, method="Newton-CG", jac=partial(objective_grad_sparce, L=L_tilde), options={"disp": False})
  eigenvector = opt.x/np.linalg.norm(opt.x)
  end = time.time()
  print(f"Elapsed time for eigenvalue calculation:{end-start}")
  return opt.fun, eigenvector
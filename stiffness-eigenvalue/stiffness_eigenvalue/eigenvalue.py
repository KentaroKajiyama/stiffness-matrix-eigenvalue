import numpy as np
import math
from scipy.optimize import minimize
from functools import partial

"""
EPS_OBJ : A small constant for objective function 
BIG_C : A big constant for optimization
"""
###########################
EPS_OBJ = 10**(-8)
BIG_C = 10**6
###########################

# 回転行列（歪対称行列）の生成
def gen_skew_symmetric(d):
  result = []
  for i in range(0, d):
    for j in range(i+1,d):
      S = np.zeros((d,d))
      S[i][j] = 1; S[j][i] = -1;
      result.append(S)
  # 平行移動用の余分なS（ただの単位行列）
  for i in range(0,d):
    result.append(np.eye(d))
  return result
# 平行移動ベクトルの生成
def gen_parallel_vector(d):
  # 回転用の余分なt（ただの0ベクトル）
  result = [np.zeros(d) for _ in range(0,math.comb(d,2))]
  [result.append(np.array([0 if i!=j else 1 for i in range(0,d)])) for j in range(0,d)]
  return result
# 基底の生成
def gen_basis(d,p):
  # 点の個数
  n = len(p)
  basis_box = []
  S_box = gen_skew_symmetric(d)
  t_box = gen_parallel_vector(d)
  for i, trans in enumerate(zip(S_box, t_box)):
    S = trans[0]; t = trans[1]; x = [];
    for j in range(n):
      x_j = S @ p[j] + t
      x.append(x_j)
    basis_box.append(np.array(x))
  return basis_box
# 目的関数
def objective(x,L):
  if np.linalg.norm(x) <= EPS_OBJ:
    return 0
  else:
    return np.dot(x.T, np.dot(L, x))/np.dot(x.T, x)
# 固有値計算（最適化による計算）。固有値と固有ベクトルを返す。
def min_non_zero_eigen(L, x0, d, p):
  # 近似行列の生成
  basis_box = gen_basis(d,p)
  L_tilde = L
  for basis in basis_box:
    L_tilde += BIG_C*np.outer(basis, basis)
  # 目的関数
  obj_func = partial(objective, L=L_tilde)
  opt = minimize(obj_func, x0)
  return opt.fun, opt.x
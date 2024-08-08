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
def gen_basis(S_box, t_box, p):
  # 点の個数
  n = len(p)
  basis_box = []
  for i, trans in enumerate(zip(S_box, t_box)):
    S = trans[0]; t = trans[1]; x = [];
    for j in range(n):
      x_j = S @ p[j] + t
      x.extend(x_j)
    # 正規化を含む
    basis_box.append(np.array(x)/np.linalg.norm(x))
  return basis_box
# 目的関数
def objective(x,L):
  if np.linalg.norm(x) <= EPS_OBJ:
    return 0
  else:
    return np.dot(x.T, np.dot(L, x))/np.dot(x.T, x)
# 固有値計算（最適化による計算）。固有値と固有ベクトルを返す。
def min_non_zero_eigen(L, x0, p, S_box, t_box):
  # 近似行列の生成
  basis_box = gen_basis(S_box, t_box, p)
  L_tilde = L
  for basis in basis_box:
    L_tilde += BIG_C*np.outer(basis, basis)
  # 目的関数
  obj_func = partial(objective, L=L_tilde)
  opt = minimize(obj_func, x0)
  return opt.fun, opt.x/np.linalg.norm(opt.x)
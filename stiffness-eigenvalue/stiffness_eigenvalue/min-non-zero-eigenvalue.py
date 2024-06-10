import numpy as np
from scipy.optimize import minimize
from functools import partial

# 極小正数
eps = 10**(-15)
# 目的関数
def objective(v,L):
  if np.norm(v) <= eps:
    return 0
  else:
    return np.dot(v.T, v) / np.dot(v.T, np.dot(L, v)) 

def non_zero_eigenvalue(L, v0):
  # 目的関数
  obj_func = partial(objective, L=L)
  # 最適化の実行
  solution = minimize(obj_func, v0)
  # 結果の表示
  print('Optimal solution:', solution.x)
  print('Objective value:', solution.fun)
  return solution.fun, solution.x


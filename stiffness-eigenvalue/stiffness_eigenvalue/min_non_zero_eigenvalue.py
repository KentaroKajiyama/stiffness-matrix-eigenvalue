import numpy as np
from scipy.optimize import minimize
from functools import partial

# 極小正数
eps = 10**(-15)
# 目的関数
def objective(v,L):
  if np.linalg.norm(v) <= eps:
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

def test_1():
  R = 10*np.random.randn(6,4)
  L = R @ R.T
  v0 = np.ones(6)/6
  eps = 1
  print("L:",L)
  print("L+eps*np.eye(6)", L+eps*np.eye(6))
  eigen_val, eigen_vec = non_zero_eigenvalue(L+eps*np.eye(6), v0)

def test_2():
  pass

if __name__ == "__main__":
  test_1()

import numpy as np
import math

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


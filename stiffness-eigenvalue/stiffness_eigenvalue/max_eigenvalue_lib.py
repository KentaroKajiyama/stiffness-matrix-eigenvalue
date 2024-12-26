import numpy as np
from scipy.sparse.linalg import eigsh
import rigidpy as rp
from stiffness_eigenvalue.framework import stiffness_matrix_sparce
from stiffness_eigenvalue.visualize import custom_visualize, plot_eigen_vals_and_alpha
import math
import time
from scipy.sparse import csr_matrix

##################################################
"""
Hyperparameter
(Ascent Direct)
EPSILON : A small constant number to make 
(Armijo Condition)
MAX_ITER_FOR_ARMIJO : The number of the max iteration for Armijo condition.
C1 : The constant for the Armijo condition
ROW : Decreasing ratio for Armijo.
(Calculation for eigenvalues)
NON_ZERO_INDEX : The index of the non zero eigenvalue of a stiffness matrix.
MAX_ITER_FOR_EIGENVALUE : The number of the iteration for calculating eigenvalues.
"""
# Ascent Direct
EPSILON = 0.05
# Armijo Condition
MAX_ITER_FOR_ARMIJO = 20
C1 = 0.7
ROW = 0.77
ALPHA = 2
# Calculation for eigenvalues
NON_ZERO_INDEX = 3
MAX_ITER_FOR_EIGENVALUE = 1500
###################################################

# ライブラリーを用いてテスト
# ライブラリを用いて直接固有値全てを計算し、最小非ゼロ固有値のみを取り出すG:k-regular graph, p:realization, eps: 近似する際のeps
"""
p : [[p(1)_1,p(1)_2,...,p(1)_d],...,[p(n)_1,p(n)_2,...,p(n)_d]] (shape: (n,d))
"""
def max_p_eigenvalue_lib(G_regular, p, visual_eigen=False):
  start = time.time()
  global NON_ZERO_INDEX
  # 各定数
  # 最初のp(realization)
  p_init = p
  # realization,alpha,固有値,固有ベクトルを記録するための箱（格納する固有値は初回を含めてMAX_ITER回分）
  p_box = []
  alpha_box = []
  eigen_val_box = []
  multiplicity_vec_box = []
  # realization p の次元 dim
  dim = p.shape[1]
  NON_ZERO_INDEX = math.comb(dim+1,2)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # pの正規化
  # p = p/np.linalg.norm(p)
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER_FOR_EIGENVALUE):
    # alphaの初期化
    alpha = ALPHA
    # realizationの格納
    p_box.append(p)
    # Armijoの条件を用いて最適な更新幅を決定する。Armijoの関数内で更新まで行って
    alpha, non_zero_smallest_eigenvalue_after, multiplicity_eigenvectors, p_after = pseudo_armijo(alpha, dim, p, bonds, G_regular)
    # alphaの値を記録
    alpha_box.append(alpha)
    # 固有値・固有ベクトルの記録
    eigen_val_box.append(non_zero_smallest_eigenvalue_after)
    # 固有値の重複度の記録
    multiplicity_vec_box.append(multiplicity_eigenvectors)
    # realizationを更新
    p = p_after
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  t = time.time()-start
  print(f"Elapsed time for eigenvalue calculation:{t}")
  # visual_eigen = Trueの場合に固有値の推移の様子をプロットする。テスト用
  if visual_eigen:
    plot_eigen_vals_and_alpha(eigen_val_box, alpha_box, multiplicity_vec_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F, label=f"opt, index={max_index} value")
    F = rp.framework(p_init, bonds)
    custom_visualize(F, label = "init")
    print("max_index:",max_index)
    print("max_eigenvalue:",eigen_val_box[max_index])
  return p_box[max_index], eigen_val_box[max_index], eigen_val_box, alpha_box, multiplicity_vec_box

# Armijoの条件、勾配の代わりに固有ベクトルを降下方向としている
"""
p : [[p(1)_1,p(1)_2,...,p(1)_d],...,[p(n)_1,p(n)_2,...,p(n)_d]] (shape: (n,d))
p has been normalized.
"""
def pseudo_armijo(alpha, dim, p, bonds, G):
  # 繰り返し回数の記録
  count = 0
  # Stiffness matrix
  L = stiffness_matrix_sparce(p,bonds)
  # Eigenvalues
  # ライブラリーの固有値計算の際に収束せず0固有値が現れない場合をカバー
  # 固有値の個数の1/3を取得してきて重複がないか後にチェックする
  while True:
    eigen_vals, eigen_vecs = eigsh(L, k=8, which='SM', tol=1e-5, ncv=20)
    non_zero_smallest_eigenvalue = eigen_vals[NON_ZERO_INDEX]
    check = eigen_vals[NON_ZERO_INDEX-1]
    if check< 1e-5:
      break
  non_zero_smallest_eigenvectors = []
  # 固有値が重複する場合にカウントする、非ゼロ固有値が0に近い場合、一緒にグルーピングされてしまう恐れあり
  matching_indices = np.where(np.isclose(eigen_vals, non_zero_smallest_eigenvalue, atol=1e-5))[0]
  if len(matching_indices) > 1:
    print(f"matching_indices:{matching_indices}")
    for i in matching_indices:
      print(f"index:{matching_indices[i]}, eigenvalue:{eigen_vals[matching_indices[i]]}")
  for index in matching_indices:
    non_zero_smallest_eigenvectors.append(eigen_vecs[:, index])
  # eigenvector
  non_zero_smallest_eigenvector = non_zero_smallest_eigenvectors[0]
  # 上昇方向の計算
  ascend_vec = ascend_dir(p, non_zero_smallest_eigenvector, dim, G)
  # 更新後のrealization p_after　あとは上昇方向にどれだけ動かすかを決定する
  p_after = p+alpha*ascend_vec
  # realizationを正規化して更新
  p_after = p_after/np.linalg.norm(p_after)
  # Stiffness matrix
  L_after = stiffness_matrix_sparce(p_after,bonds)
  # 更新後の固有値計算
  # 固有値が重複する場合にカウントする、非ゼロ固有値が0に近い場合、一緒にグルーピングされてしまう恐れあり
  while True:
    # Eigenvalues
    eigen_vals_after, eigen_vecs_after = eigsh(L_after, NON_ZERO_INDEX+1, which='SM', tol=1e-5, ncv=20)
    # d=2 4th minimum eigenvalue
    non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
    check = eigen_vals_after[NON_ZERO_INDEX-1]
    if check< 1e-7:
      break
  # non_zero_smallest_eigenvectorを上昇方向としているため、これを勾配と見てArmijo条件を適用する。
  cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(np.ravel(ascend_vec), np.ravel(ascend_vec)) - non_zero_smallest_eigenvalue_after
  while(cd>0 and count < MAX_ITER_FOR_ARMIJO):
    # alphaの更新。1次減少ではなく、別の割合で更新する方法も考えられる
    alpha *= ROW 
    # realizationを正規化して更新
    p_after = p+alpha*ascend_vec
    p_after = p_after/np.linalg.norm(p_after)
    # Stiffness matrix
    L_after = stiffness_matrix_sparce(p_after, bonds)
    while True:
      # Eigenvalues
      eigen_vals_after, eigen_vecs_after = eigsh(L_after, NON_ZERO_INDEX+1, which='SM', tol=1e-5, ncv=20)
      # d=2 4th minimum eigenvalue
      non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
      check = eigen_vals_after[NON_ZERO_INDEX-1]
      if check< 1e-7:
        break
    cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(np.ravel(ascend_vec), np.ravel(ascend_vec)) - non_zero_smallest_eigenvalue_after
    count +=1
  return alpha, non_zero_smallest_eigenvalue_after,len(non_zero_smallest_eigenvectors), p_after

# 上昇方向の決定関数（最大化問題なので上昇方向であることに注意）TODO:中身があっているかもう一度確認する。
"""
p : [[p(1)_1,p(1)_2,...,p(1)_d],...,[p(n)_1,p(n)_2,...,p(n)_d]] (shape: (n,d))
eigenvector : eigenvector of the stiffness matrix
              [v(1)_1,v(1)_2,...,v(1)_d,...,v(n)_1,v(n)_2,...,v(n)_d] (shape: (n*d,))
p has been normalized.

output: ascend_vec : [[ascend_vec(1)_1,ascend_vec(1)_2,...,ascend_vec(1)_d],...,[ascend_vec(n)_1,ascend_vec(n)_2,...,ascend_vec(n)_d]] (shape: (n,d))
"""
def ascend_dir(p, eigenvector, dim, G):
  n = p.shape[0]
  ascend_vec = np.zeros((n,dim))
  for k in range(n):
    for l in range(dim):
      A = []
      row_indices = []
      col_indices = []
      # L_dot(ik) 
      # if row = dim*i+l, col = dim*k+l, then, 4(p(k)-p(i))_l, 
      # elif row = dim*i+s, col = dim*k+l, then 2(p(k)-p(i))_s, 
      # elif row = dim*i+l, col = dim*k+t, then, 2(p(k)-p(i))_t 
      for i in range(0,n):
        if i == k:
          continue
        A.append(4*(p[k][l]-p[i][l]))
        row_indices.append(dim*i+l)
        col_indices.append(dim*k+l)
        for s in range(dim):
          if s == k:
            continue
          A.append(2*(p[k][s]-p[i][s]))
          row_indices.append(dim*i+s)
          col_indices.append(dim*k+l)
        for t in range(dim):
          if t == l:
            continue
          A.append(2*(p[k][t]-p[i][t]))
          row_indices.append(dim*i+l)
          col_indices.append(dim*k+t)
      # L_dot(ki) 
      # if row = dim*k+l, col = dim*i+l, then, 4(p(k)-p(i))_l, 
      # elif row = dim*k+s, col = dim*i+l, then 2(p(k)-p(i))_s, 
      # elif row = dim*k+l, col = dim*i+t, then, 2(p(k)-p(i))_t 
      for i in range(0,n):
        if i == k:
          continue
        A.append(4*(p[k][l]-p[i][l]))
        row_indices.append(dim*k+l)
        col_indices.append(dim*i+l)
        for s in range(dim):
          if s == k:
            continue
          A.append(2*(p[k][s]-p[i][s]))
          row_indices.append(dim*k+s)
          col_indices.append(dim*i+l)
        for t in range(dim):
          if t == l:
            continue
          A.append(2*(p[k][t]-p[i][t]))
          row_indices.append(dim*k+l)
          col_indices.append(dim*i+t)
      # L_dot(ii) where i in N_G(k) 
      # if row = col = dim*i+l, then, 2(p(k)-p(i))_l, 
      # elif row = dim*i+s, col = dim*i+l, then, (p(k)-p(i))_s, 
      # elif row = dim*i+l, col = dim*i+t, then, (p(k)-p(i))_t
      for i in G.neighbors(k):
        A.append(2*(p[k][l]-p[i][l]))
        row_indices.append(dim*i+l)
        col_indices.append(dim*i+l)
        for s in range(dim):
          if s == l:
            continue
          A.append(p[k][s]-p[i][s])
          row_indices.append(dim*i+s)
          col_indices.append(dim*i+l)
        for t in range(dim):
          if t == l:
            continue
          A.append(p[k][t]-p[i][t])
          row_indices.append(dim*i+l)
          col_indices.append(dim*i+t)
      # L_dot(kk) 
      # if row = col = dim*k+l, then, sum of 2(p(k)-p(j))_l over j in N_G(k), 
      # elif row = dim*k+s, col = dim*k+l, then, sum of (p(k)-p(j))_s over j in N_G(k), 
      # elif row = dim*k+l, col = dim*k+t, then, sum of (p(k)-p(j))_t over j in N_G(k)
      # d(k-1)列目からdk-1列目まで
      A.append(sum([2*(p[k][l]-p[j][l]) for j in G.neighbors(k)]))
      row_indices.append(dim*k+l)
      col_indices.append(dim*k+l)
      for s in range(dim):
        if s == l:
          continue
        A.append(sum([p[k][s]-p[j][s] for j in G.neighbors(k)]))
        row_indices.append(dim*k+s)
        col_indices.append(dim*k+l)
      for t in range(dim):
        if t == l:
          continue
        A.append(sum([p[k][t]-p[j][t] for j in G.neighbors(k)]))
        row_indices.append(dim*k+l)
        col_indices.append(dim*k+t)
      # Generate the sparse matrix
      diff_L = csr_matrix((A, (row_indices, col_indices)), shape=(dim*n, dim*n))
      # 上昇方向の計算(p_d(k-1)+lによる微分)
      ascend_vec_element = np.dot(eigenvector, diff_L.dot(eigenvector))
      ascend_vec[k][l] = ascend_vec_element
  return ascend_vec

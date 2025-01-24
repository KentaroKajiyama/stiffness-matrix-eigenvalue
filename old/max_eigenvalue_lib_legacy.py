import numpy as np
from scipy.sparse.linalg import eigsh
import rigidpy as rp
from stiffness_eigenvalue.framework import stiffness_matrix_sparce
from stiffness_eigenvalue.visualize import custom_visualize, plot_eigen_vals_and_alpha
import math
import time

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
EPSILON = 0.01
# Armijo Condition
MAX_ITER_FOR_ARMIJO = 17
C1 = 0.10
ROW = 0.8
ALPHA = 5

# Calculation for eigenvalues
NON_ZERO_INDEX = 3
MAX_ITER_FOR_EIGENVALUE = 1000
CHANGE_HUGE = 3000
SWITCH = 300
###################################################

# 上昇方向の決定関数（最大化問題なので上昇方向であることに注意）
def pseudo_ascent_dir(p, bonds, eigenvectors, dim):
  eigen_val_box = []
  eigen_vec_box = []
  for eigenvector in eigenvectors:
    # +の場合(eigenvectorと同じ方向に移動させる場合)
    p_plus = p + EPSILON*eigenvector.reshape(-1, dim)
    # Stiffness matrix
    L_plus = stiffness_matrix_sparce(p_plus,bonds)
    # -の場合(eigenvectorと同じ方向に移動させる場合)
    p_minus = p - EPSILON*eigenvector.reshape(-1, dim)
    # Stiffness matrix
    L_minus = stiffness_matrix_sparce(p_minus,bonds)
    # Eigenvalues
    eigen_vals_plus, _ = eigsh(L_plus, NON_ZERO_INDEX+1, which='SM')
    eigen_vals_minus, _ = eigsh(L_minus, NON_ZERO_INDEX+1, which='SM')
    eigen_val_box.append(eigen_vals_plus[NON_ZERO_INDEX])
    eigen_val_box.append(eigen_vals_minus[NON_ZERO_INDEX])
    eigen_vec_box.append(eigenvector)
    eigen_vec_box.append(-eigenvector)
  max_index = np.argmax(eigen_val_box)
  return eigen_vec_box[max_index]
# Armijoの条件、勾配の代わりに固有ベクトルを降下方向としている
def pseudo_armijo(alpha, dim, p, bonds, explosion_count_plus, explosion_count_minus, flag, explosion_switch):
  # 繰り返し回数の記録
  count = 0
  flag = 0
  # Stiffness matrix
  L = stiffness_matrix_sparce(p,bonds)
  # Eigenvalues
  eigen_vals, eigen_vecs = eigsh(L, k=L.shape[0]/4, which='SM')
  # d=2 4th non-zero minimum eigenvalue
  non_zero_smallest_eigenvalue = eigen_vals[NON_ZERO_INDEX]
  matching_indices = np.where(np.isclose(eigen_vals, non_zero_smallest_eigenvalue, atol=1e-6))
  non_zero_smallest_eigenvectors = eigen_vecs[:, matching_indices]
  # eigenvalueの方向決め、これによりnon_zero_smallest_eigenvectorが上昇方向として確定する。 
  non_zero_smallest_eigenvector = pseudo_ascent_dir(p, bonds, non_zero_smallest_eigenvectors, dim)
  # 更新後のrealization p_after
  p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_after = stiffness_matrix_sparce(p_after,bonds)
  # Eigenvalues
  eigen_vals_after, eigen_vecs_after = eigsh(L_after, NON_ZERO_INDEX+1, which='SM')
  # d=2 4th minimum eigenvalue
  non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
  # non_zero_smallest_eigenvectorを上昇方向としているため、これを勾配と見てArmijo条件を適用する。（向きを調整する前のもともとの固有ベクトルに戻すためにddをかけている）＜－ここのddいる？？いらなそう
  cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
  while(cd>0 and count < MAX_ITER_FOR_ARMIJO):
    alpha *= ROW # if count > 0 else alpha = alpha # 二次減少, 変えるときは大きく、変えないときは小さく。
    # alpha = np.exp(-count**1.6/10)*10
    p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1,dim)
    # Stiffness matrix
    L_after = stiffness_matrix_sparce(p_after, bonds)
    # Eigenvalues
    eigen_vals_after, eigen_vecs_after = eigsh(L_after, NON_ZERO_INDEX+1, which='SM')
    # d=2 4th minimum eigenvalue
    non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
    cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
    count +=1
  # print(f"cd: {cd}, alpha: {alpha}, non_zero_smallest_eigenvalue: {non_zero_smallest_eigenvalue}, non_zero_smallest_eigenvalue_after: {non_zero_smallest_eigenvalue_after}")
  # if cd <= 0:
  #   explosion_count_minus += 1
  # else:
  #   explosion_count_plus += 1
  if explosion_count_plus > 500:
    explosion_count_plus = 0
    explosion_count_minus = 0
    flag = 1
  if explosion_count_minus > 250 or explosion_switch == 1:
    explosion_count_minus = 0
    explosion_count_plus = 0
    flag = -1
    explosion_switch = 0
    alpha = 0.2
    # 更新後のrealization p_after
    p_after = p-alpha*non_zero_smallest_eigenvector.reshape(-1, dim)
    # Stiffness matrix
    L_after = stiffness_matrix_sparce(p_after,bonds)
    # Eigenvalues
    eigen_vals_after, eigen_vecs_after = eigsh(L_after, NON_ZERO_INDEX+1, which='SM')
    # d=2 4th minimum eigenvalue
    non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
  return alpha, non_zero_smallest_eigenvalue_after, p_after, explosion_count_plus, explosion_count_minus, flag, explosion_switch
# ライブラリーを用いてテスト
# ライブラリを用いて直接固有値全てを計算し、最小非ゼロ固有値のみを取り出すG:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue_lib_legacy(G_regular, p, eigen_vec_0, visual_eigen=False):
  start = time.time()
  # 各定数
  # 最初のp(realization)
  explosion_count_plus = 0
  explosion_count_minus = 0
  flag = 0
  explosion_switch = 0
  explosion_diff = 0
  switch_count = 0
  # realization,alpha,固有値,固有ベクトルを記録するための箱（格納する固有値は初回を含めてMAX_ITER回分）
  p_box = []
  alpha_box = []
  eigen_val_box = []
  eigen_vec_box = []
  # 次元 dim
  dim = p.shape[1]
  NON_ZERO_INDEX = math.comb(dim+1,2)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # pの正規化
  p = p/np.linalg.norm(p)
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER_FOR_EIGENVALUE):
    # 移動分の係数
    if i < CHANGE_HUGE:
      alpha = ALPHA
    else:
      alpha = ALPHA/10
      explosion_count_minus = -100
      switch_count = 0
    # realizationの格納
    p_box.append(p)
    if i > 1:
      explosion_diff += eigen_val_box[-1] - eigen_val_box[-2]
      if explosion_diff > 0.5:
        print(f"{i}th iteration: explosion_diff:{explosion_diff}")
        alpha = 0.1
        explosion_diff = 0
      switch_count += 1
    if switch_count == SWITCH:
      print(f"explosion_diff:{explosion_diff}")
      if explosion_diff < 0.2:
        explosion_switch = 1
        print(f"{i}th iteration: switch explosion")
      switch_count = 0
      explosion_diff = 0
    # Armijoの条件を用いて最適な更新幅を決定する。Armijoの関数内で更新まで行っている
    alpha, non_zero_smallest_eigenvalue_after, p_after, explosion_count_plus, explosion_count_minus, flag, explosion_switch = pseudo_armijo(alpha, dim, p, bonds, explosion_count_plus, explosion_count_minus, flag, explosion_switch)
    if flag == 1:
      print(f"{i}th iteration: plus explosion")
    elif flag == -1:
      print(f"{i}th iteration: minus explosion")
    # alphaの値を記録
    alpha_box.append(alpha)
    # realizationを正規化して更新
    p = p_after/np.linalg.norm(p_after)
    # 固有値・固有ベクトルの記録
    eigen_val_box.append(non_zero_smallest_eigenvalue_after)
    # eigen_vec_box.append(non_zero_smallest_eigenvector_after)
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  t = time.time()-start
  print(f"Elapsed time for eigenvalue calculation:{t}")
  # visual_eigen = Trueの場合に固有値の推移の様子をプロットする。テスト用
  if visual_eigen:
    plot_eigen_vals_and_alpha(eigen_val_box, alpha_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F, label=f"opt, index={max_index} value")
    # F = rp.framework(p_init, bonds)
    # custom_visualize(F, label = "init")
    print("max_index:",max_index)
    print("max_eigenvalue:",eigen_val_box[max_index])
  return p_box[max_index], eigen_val_box[max_index], eigen_vec_box[max_index]

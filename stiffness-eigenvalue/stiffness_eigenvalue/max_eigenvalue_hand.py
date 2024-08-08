import numpy as np
import rigidpy as rp
from stiffness_eigenvalue.framework import stiffness_matrix
from stiffness_eigenvalue.visualize import custom_visualize, plot_eigen_vals,plot_alpha
from stiffness_eigenvalue.eigenvalue import gen_parallel_vector, gen_skew_symmetric, min_non_zero_eigen

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
EPSILON = 0.0000001
# Armijo Condition
MAX_ITER_FOR_ARMIJO = 13
C1 = 0.10
ROW = 0.6
# Calculation for eigenvalues
NON_ZERO_INDEX = 3
MAX_ITER_FOR_EIGENVALUE = 3000
###################################################

# 上昇方向の決定関数（最大化問題なので上昇方向であることに注意）
def pseudo_ascent_dir(p, bonds, eigenvalue_current, eigenvector, dim, S_box, t_box):
  # +の場合(eigenvectorと同じ方向に移動させる場合)
  p_plus = p + EPSILON*eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_plus = stiffness_matrix(p_plus,bonds)
  # Eigenvalue
  eigenvalue_plus, _ = min_non_zero_eigen(L_plus, eigenvector, p_plus, S_box, t_box)
  # # -の場合(eigenvectorと逆方向に移動させる場合)（-の場合は省略して現在の値と比較する）
  # p_minus = p - EPSILON*eigenvector.reshape(-1, dim)
  # # Stiffness matrix
  # L_minus = stiffness_matrix(p_minus,bonds)
  # # Eigenvalues
  # eigen_vals_minus, _ = min_non_zero_eigen(L_minus, eigenvector, p_minus, S_box, t_box)
  if eigenvalue_plus >= eigenvalue_current:
    return 1
  else:
    return -1
# Armijoの条件、勾配の代わりに固有ベクトルを降下方向としている
def pseudo_armijo(alpha, eigen_vec_0, dim, p, bonds, S_box, t_box):
  # 繰り返し回数の記録
  count = 0
  # Stiffness matrix
  L = stiffness_matrix(p,bonds)
  # Eigenvalues, Eigenvectors
  non_zero_smallest_eigenvalue, non_zero_smallest_eigenvector = min_non_zero_eigen(L, eigen_vec_0, p, S_box, t_box)
  # eigenvalueの方向決め、これによりnon_zero_smallest_eigenvectorが上昇方向として確定する。
  dd = pseudo_ascent_dir(p, bonds, non_zero_smallest_eigenvalue, non_zero_smallest_eigenvector, dim, S_box, t_box)
  non_zero_smallest_eigenvector *= dd
  # 更新後のrealization p_after
  p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_after = stiffness_matrix(p_after,bonds)
  # Eigenvalues, Eigenvectors
  non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after = min_non_zero_eigen(L_after, non_zero_smallest_eigenvector, p_after, S_box, t_box)
  # non_zero_smallest_eigenvectorを上昇方向としているため、これを勾配と見てArmijo条件を適用する。（向きを調整する前のもともとの固有ベクトルに戻すためにddをかけている）
  cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, dd*non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
  while(cd>0 and count < MAX_ITER_FOR_ARMIJO):
    alpha *= ROW
    p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1,dim)
    # Stiffness matrix
    L_after = stiffness_matrix(p_after, bonds)
    # Eigenvalues, Eigenvectors
    non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after = min_non_zero_eigen(L_after, non_zero_smallest_eigenvector, p_after, S_box, t_box)
    cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
    count +=1
  return alpha, non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after, p_after
# ライブラリーを用いてテスト
# ライブラリを用いて直接固有値全てを計算し、最小非ゼロ固有値のみを取り出すG:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue_hand(G_regular, p, eigen_vec_0, visual_eigen=False):
  # 各定数
  # 最初のp(realization)
  p_init = p
  # realization,alpha,固有値,固有ベクトルを記録するための箱（格納する固有値は初回を含めてMAX_ITER回分）
  p_box = []
  alpha_box = []
  eigen_val_box = []
  eigen_vec_box = []
  # 次元 dim
  dim = p.shape[1]
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # 基底生成用のSとt
  S_box = gen_skew_symmetric(dim)
  t_box = gen_parallel_vector(dim)
  # pの正規化
  p = p/np.linalg.norm(p)
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for _ in range(MAX_ITER_FOR_EIGENVALUE):
    # realizationの格納
    p_box.append(p)
    # 移動分の係数
    alpha = 1
    # Armijoの条件を用いて最適な更新幅を決定する。Armijoの関数内で更新まで行っている
    alpha, non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after, p_after = pseudo_armijo(alpha, eigen_vec_0, dim, p, bonds, S_box, t_box)
    # alphaの値を記録
    alpha_box.append(alpha)
    # realizationを正規化して更新
    p = p_after/np.linalg.norm(p_after)
    # 固有値・固有ベクトルの記録
    eigen_val_box.append(non_zero_smallest_eigenvalue_after)
    eigen_vec_box.append(non_zero_smallest_eigenvector_after)
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  # visual_eigen = Trueの場合に固有値の推移の様子をプロットする。テスト用
  if visual_eigen:
    plot_eigen_vals(eigen_val_box)
    plot_alpha(alpha_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F, label=f"opt, index={max_index} value")
    F = rp.framework(p_init, bonds)
    custom_visualize(F, label = "init")
    print(f"max index: {max_index}")
    print(f"sup non zero eigenvalue: {eigen_val_box[max_index]}")
  return p_box[max_index], eigen_val_box[max_index], eigen_vec_box[max_index]

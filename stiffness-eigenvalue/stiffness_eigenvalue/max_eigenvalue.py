import numpy as np
import rigidpy as rp
from stiffness_eigenvalue.framework import stiffness_matrix
from stiffness_eigenvalue.visualize import custom_visualize, plot_eigen_vals,plot_alpha, on_key

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
"""
# Ascent Direct
EPSILON = 0.0000001
# Armijo Condition
MAX_ITER_FOR_ARMIJO = 2
C1 = 0.1
ROW = 0.6
# Calculation for eigenvalues
NON_ZERO_INDEX = 3

###################################################

# 上昇方向の決定関数（最大化問題なので上昇方向であることに注意）
def pseudo_ascent_dir(p, bonds, eigenvector, dim):
  # +の場合(eigenvectorと同じ方向に移動させる場合)
  p_plus = p + EPSILON*eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_plus = stiffness_matrix(p_plus,bonds)
  # Eigenvalues
  eigen_vals_plus = np.linalg.eig(L_plus)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_plus = np.argsort(eigen_vals_plus)
  eigen_vals_plus = eigen_vals_plus[sorted_indices_plus]
  # -の場合(eigenvectorと逆方向に移動させる場合)
  p_minus = p - EPSILON*eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_minus = stiffness_matrix(p_minus,bonds)
  # Eigenvalues
  eigen_vals_minus = np.linalg.eig(L_minus)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_minus = np.argsort(eigen_vals_minus)
  eigen_vals_minus = eigen_vals_minus[sorted_indices_minus]
  if eigen_vals_plus[NON_ZERO_INDEX] >= eigen_vals_minus[NON_ZERO_INDEX]:
    return 1
  else:
    return -1
# Armijoの条件、勾配の代わりに固有ベクトルを降下方向としている
def pseudo_armijo(alpha, dim, p, bonds):
  # 繰り返し回数の記録
  count = 0
  # Stiffness matrix
  L = stiffness_matrix(p,bonds)
  # Eigenvalues
  eigen_vals, eigen_vecs = np.linalg.eig(L)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices = np.argsort(eigen_vals)
  eigen_vals, eigen_vecs = eigen_vals[sorted_indices], eigen_vecs[sorted_indices]
  # d=2 4th minimum eigenvalue
  non_zero_smallest_eigenvalue = eigen_vals[NON_ZERO_INDEX]
  non_zero_smallest_eigenvector = eigen_vecs[:,NON_ZERO_INDEX].real
  # eigenvalueの方向決め、これによりnon_zero_smallest_eigenvectorが上昇方向として確定する。
  dd = pseudo_ascent_dir(p, bonds, non_zero_smallest_eigenvector, dim)
  non_zero_smallest_eigenvector *= dd
  # 更新後のrealization p_after
  p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1, dim)
  # Stiffness matrix
  L_after = stiffness_matrix(p_after,bonds)
  # Eigenvalues
  eigen_vals_after, eigen_vecs_after = np.linalg.eig(L_after)
  # Sort eigenvalues and eigenvectors in the decending order.
  sorted_indices_after = np.argsort(eigen_vals_after)
  eigen_vals_after, eigen_vecs_after = eigen_vals_after[sorted_indices_after], eigen_vecs_after[sorted_indices_after]
  # d=2 4th minimum eigenvalue
  non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after = eigen_vals_after[NON_ZERO_INDEX], eigen_vecs_after[:,NON_ZERO_INDEX]
  # non_zero_smallest_eigenvectorを上昇方向としているため、これを勾配と見てArmijo条件を適用する。
  cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
  while(cd>0 and count < MAX_ITER_FOR_ARMIJO):
    alpha *= ROW
    p_after = p+alpha*non_zero_smallest_eigenvector.reshape(-1,dim)
    # Stiffness matrix
    L_after = stiffness_matrix(p_after, bonds)
    # Eigenvalues
    eigen_vals_after, eigen_vecs_after = np.linalg.eig(L_after)
    # Sort eigenvalues and eigenvectors in the decending order.
    sorted_indices_after = np.argsort(eigen_vals_after)
    eigen_vals_after, eigen_vecs_after = eigen_vals_after[sorted_indices_after], eigen_vecs_after[sorted_indices_after]
    # d=2 4th minimum eigenvalue
    non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
    non_zero_smallest_eigenvector_after = eigen_vecs_after[:,NON_ZERO_INDEX].real
    cd = non_zero_smallest_eigenvalue + C1*alpha*np.dot(non_zero_smallest_eigenvector, non_zero_smallest_eigenvector) - non_zero_smallest_eigenvalue_after
    count +=1
  return alpha, non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after, p_after
# ライブラリーを用いてテスト
# ライブラリを用いて直接固有値全てを計算し、最小非ゼロ固有値のみを取り出すG:k-regular graph, p:realization, eps: 近似する際のeps
def max_p_eigenvalue_lib(G_regular, p, eigen_vec_0, visual_eigen=False):
  # 各定数
  # 最初のp(realization)
  p_init = p
  # 最大繰り返し回数この分だけ固有ベクトル方向に移動させる
  MAX_ITER = 3000
  # realization,alpha,固有値,固有ベクトルを記録するための箱（格納する固有値は初回を含めてMAX_ITER回分）
  p_box = []
  alpha_box = []
  eigen_val_box = []
  eigen_vec_box = []
  # 次元 dim
  dim = p.shape[1]
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # pの正規化
  p = p/np.linalg.norm(p)
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for _ in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    # 移動分の係数
    alpha = 1
    # Armijoの条件を用いて最適な更新幅を決定する。Armijoの関数内で更新まで行っている
    alpha, non_zero_smallest_eigenvalue_after, non_zero_smallest_eigenvector_after, p_after = pseudo_armijo(alpha, dim, p, bonds)
    # alphaの値を記録
    alpha_box.append(alpha)
    # realizationを正規化して更新
    p = p_after/np.linalg.norm(p_after)
    # 固有値・固有ベクトルの記録
    eigen_val_box.append(non_zero_smallest_eigenvalue_after)
    eigen_vec_box.append(non_zero_smallest_eigenvector_after)
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_val_box)
  # visual_eigen = Trueの場合に固有値の推移の様子をプロットする。
  if visual_eigen:
    plot_eigen_vals(eigen_val_box)
    plot_alpha(alpha_box)
    F = rp.framework(p_box[max_index], bonds)
    custom_visualize(F, label=f"opt, index={max_index} value")
    F = rp.framework(p_init, bonds)
    custom_visualize(F, label = "init")
  return p_box[max_index], eigen_val_box[max_index], eigen_vec_box[max_index]

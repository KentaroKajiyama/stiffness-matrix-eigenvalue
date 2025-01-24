import numpy as np
from scipy.sparse.linalg import eigsh
import rigidpy as rp
from framework import stiffness_matrix_sparce
from visualize import custom_visualize, plot_eigen_vals_and_alpha
import math
import time
from dotenv import load_dotenv
import os
from tqdm import tqdm
from rich.progress import track
import signal


load_dotenv("config/.env")


##################################################
"""
Hyperparameter
(Armijo Condition)
MAX_ITER_FOR_ARMIJO : The number of the max iteration for Armijo condition.
C1 : The constant for the Armijo condition
ROW : Decreasing ratio for Armijo.
(Calculation for eigenvalues)
NON_ZERO_INDEX : The index of the non zero eigenvalue of a stiffness matrix.
MAX_ITER_FOR_EIGENVALUE : The number of the iteration for calculating eigenvalues.
"""
# Armijo Condition
MAX_ITER_FOR_ARMIJO = int(os.getenv("MAX_ITER_FOR_ARMIJO"))
C1 = float(os.getenv("C1"))
ROW = float(os.getenv("ROW"))
ALPHA = float(os.getenv("ALPHA"))
# Calculation for eigenvalues
NON_ZERO_INDEX = int(os.getenv("NON_ZERO_INDEX"))
MAX_ITER_FOR_EIGENVALUE = int(os.getenv("MAX_ITER_FOR_EIGENVALUE"))
###################################################

# ライブラリを用いて固有値をNON_ZERO_INDEX+1個計算し、最小非ゼロ固有値のみを取り出す G:k-regular graph, p:realization
"""
p : [[p(1)_1,p(1)_2,...,p(1)_d],...,[p(n)_1,p(n)_2,...,p(n)_d]] (shape: (n,d))
"""
def max_p_eigenvalue_lib(G_regular, p, visual_eigen=False, max_iter_for_armijo=None):
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
  for i in track(range(MAX_ITER_FOR_EIGENVALUE), description="Calculating eigenvalues", total=MAX_ITER_FOR_ARMIJO):
    # alphaの初期化
    alpha = ALPHA
    # realizationの格納
    p_box.append(p)
    # Armijoの条件を用いて最適な更新幅を決定する。Armijoの関数内で更新まで行って
    alpha, non_zero_smallest_eigenvalue_after, multiplicity_eigenvectors, p_after = armijo(alpha, p, bonds, G_regular, max_iter_for_armijo=max_iter_for_armijo, index=i)
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
  print(f"Elapsed time for eigenvalue calculation:{t/60:.1f} [min]")
  print("max_index:",max_index)
  print("max_eigenvalue:",eigen_val_box[max_index])
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
def timeout_handler(signum, frame):
  raise TimeoutError

def armijo(alpha, p, bonds, G, max_iter_for_armijo=None, timeout_sec=60):
  if max_iter_for_armijo:
    MAX_ITER_FOR_ARMIJO = max_iter_for_armijo
  
  count = 0
  p_after = None  # p_afterが一度も計算されなかった場合の対応
  
  try:
    # Stiffness matrix
    L = stiffness_matrix_sparce(p, bonds)
    
    # Timeout 設定
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)  # タイムアウト設定
    
    while True:
      try:
        eigen_vals, eigen_vecs = eigsh(L, k=8, which='SM', tol=1e-5, ncv=20)
        signal.alarm(0)  # タイムアウト解除
      except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        raise TimeoutError
      
      non_zero_smallest_eigenvalue = eigen_vals[NON_ZERO_INDEX]
      check = eigen_vals[NON_ZERO_INDEX-1]
      if check < 1e-5:
        break
    
    non_zero_smallest_eigenvectors = []
    matching_indices = np.where(np.isclose(eigen_vals, non_zero_smallest_eigenvalue, atol=1e-2))[0]
    for index in matching_indices:
      non_zero_smallest_eigenvectors.append(eigen_vecs[:, index])
    
    non_zero_smallest_eigenvector = non_zero_smallest_eigenvectors[0]
    ascend_vec = ascend_dir(p, non_zero_smallest_eigenvector, G)
    p_after = p + alpha * ascend_vec
    p_after /= np.linalg.norm(p_after)
    
    L_after = stiffness_matrix_sparce(p_after, bonds)
    
    signal.alarm(timeout_sec)  # タイムアウト設定
    while True:
      try:
        eigen_vals_after, _ = eigsh(L_after, NON_ZERO_INDEX+1, which='SM', tol=1e-5, ncv=20)
        signal.alarm(0)  # タイムアウト解除
      except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        raise TimeoutError
      
      non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
      check = eigen_vals_after[NON_ZERO_INDEX-1]
      if check < 1e-7:
        break
    
    cd = non_zero_smallest_eigenvalue + C1 * alpha * np.dot(np.ravel(ascend_vec), np.ravel(ascend_vec)) - non_zero_smallest_eigenvalue_after
    while cd > 0 and count < MAX_ITER_FOR_ARMIJO:
      alpha *= ROW
      p_after = p + alpha * ascend_vec
      p_after /= np.linalg.norm(p_after)
      L_after = stiffness_matrix_sparce(p_after, bonds)
      
      signal.alarm(timeout_sec)  # タイムアウト設定
      while True:
        try:
          eigen_vals_after, _ = eigsh(L_after, NON_ZERO_INDEX+1, which='SM', tol=1e-5, ncv=20)
          signal.alarm(0)  # タイムアウト解除
        except Exception as e:
          print(f"Eigenvalue computation failed: {e}")
          raise TimeoutError
        
        non_zero_smallest_eigenvalue_after = eigen_vals_after[NON_ZERO_INDEX]
        check = eigen_vals_after[NON_ZERO_INDEX-1]
        if check < 1e-7:
          break
      
      cd = non_zero_smallest_eigenvalue + C1 * alpha * np.dot(np.ravel(ascend_vec), np.ravel(ascend_vec)) - non_zero_smallest_eigenvalue_after
      count += 1
    
  except TimeoutError:
    print("Timeout occurred or eigenvalue computation failed.")
    alpha = 0
    non_zero_smallest_eigenvalue_after = 0
    non_zero_smallest_eigenvectors = []
    if p_after is None:
      p_after = 2 * p  # 一度も計算されていない場合
  
  return alpha, non_zero_smallest_eigenvalue_after, len(non_zero_smallest_eigenvectors), p_after

# 上昇方向の決定関数（最大化問題なので上昇方向であることに注意）
"""
p : [[p(1)_1,p(1)_2,...,p(1)_d],...,[p(n)_1,p(n)_2,...,p(n)_d]] (shape: (n,d))
eigenvector : eigenvector of the stiffness matrix
              [v(1)_1,v(1)_2,...,v(1)_d,...,v(n)_1,v(n)_2,...,v(n)_d] (shape: (n*d,))
p has been normalized.

output: ascend_vec : [[ascend_vec(1)_1,ascend_vec(1)_2,...,ascend_vec(1)_d],...,[ascend_vec(n)_1,ascend_vec(n)_2,...,ascend_vec(n)_d]] (shape: (n,d))
"""
def ascend_dir(p, x, G):
  """
  p: 座標情報 (n x dim)
  x: 固有ベクトル (次元は dim*n だと仮定)
  dim: 次元
  G: ネットワーク (neighbors(k) で頂点kの近傍を返す)

  戻り値
  -------
  ascend_vec : (n x dim) のnumpy配列
  """
  n = p.shape[0]
  dim = p.shape[1]
  ascend_vec = np.zeros((n, dim), dtype=np.float64)

  # x_i に対応する index は行列L の (i座標, s座標) を 1次元化したもの:
  # たとえば i行目に対応するものは [dim*i, dim*i+1, ..., dim*i + (dim-1)]
  # という対応づけを想定
  # xは (dim*n,) なので、x[dim*i + s] が「頂点 i の次元 s 成分」に対応

  for k in range(n):
    # k番目頂点に対して
    neighbors_k = list(G.neighbors(k))

    for l in range(dim):
      # (k,l)に対する微分
      val_kl = 0.0  # ここに x^T (∂L/∂p[k][l]) x を累積

      # ==========================
      # 1) L_dot(ik) の寄与 ただし、i \in N_G(k)
      # ==========================
      for i in neighbors_k:
        # -- (row, col) = (dim*i + l, dim*k + l)
        dL_ik_ll = 4*(p[k][l] - p[i][l])
        val_kl += dL_ik_ll * x[dim*i + l] * x[dim*k + l]

        # -- (row, col) = (dim*i + s, dim*k + l)
        for s in range(dim):
          if s == l:
            continue
          dL_ik_sl = 2*(p[k][s] - p[i][s])
          val_kl += dL_ik_sl * x[dim*i + s] * x[dim*k + l]

        # -- (row, col) = (dim*i + l, dim*k + t)
        for t in range(dim):
          if t == l:
            continue
          dL_ik_lt = 2*(p[k][t] - p[i][t])
          val_kl += dL_ik_lt * x[dim*i + l] * x[dim*k + t]

      # ==========================
      # 2) L_dot(ki) の寄与 ただし、i \in N_G(k)
      # ==========================
      for i in neighbors_k:
        # -- (row, col) = (dim*k + l, dim*i + l)
        dL_ki_ll = 4*(p[k][l] - p[i][l])
        val_kl += dL_ki_ll * x[dim*k + l] * x[dim*i + l]

        # -- (row, col) = (dim*k + s, dim*i + l)
        for s in range(dim):
          if s == l:
            continue
          dL_ki_sl = 2*(p[k][s] - p[i][s])
          val_kl += dL_ki_sl * x[dim*k + s] * x[dim*i + l]

        # -- (row, col) = (dim*k + l, dim*i + t)
        for t in range(dim):
          if t == l:
            continue
          dL_ki_lt = 2*(p[k][t] - p[i][t])
          val_kl += dL_ki_lt * x[dim*k + l] * x[dim*i + t]

      # ==========================
      # 3) L_dot(ii) の寄与
      #    i in N_G(k)
      # ==========================
      for i in neighbors_k:
        # (row=col=dim*i + l)
        dL_ii_ll = 2*(p[k][l] - p[i][l])
        val_kl += dL_ii_ll * x[dim*i + l] * x[dim*i + l]

        # (row=dim*i + s, col=dim*i + l)
        for s in range(dim):
          if s == l:
            continue
          dL_ii_sl = (p[k][s] - p[i][s])
          val_kl += dL_ii_sl * x[dim*i + s] * x[dim*i + l]

        # (row=dim*i + l, col=dim*i + t)
        for t in range(dim):
          if t == l:
            continue
          dL_ii_lt = (p[k][t] - p[i][t])
          val_kl += dL_ii_lt * x[dim*i + l] * x[dim*i + t]

      # ==========================
      # 4) L_dot(kk) の寄与
      # ==========================
      # (row=col=dim*k + l)
      dL_kk_ll = sum([2*(p[k][l] - p[j][l]) for j in neighbors_k])
      val_kl += dL_kk_ll * x[dim*k + l] * x[dim*k + l]

      # (row=dim*k + s, col=dim*k + l)
      for s in range(dim):
        if s == l:
          continue
        dL_kk_sl = sum([p[k][s] - p[j][s] for j in neighbors_k])
        val_kl += dL_kk_sl * x[dim*k + s] * x[dim*k + l]

      # (row=dim*k + l, col=dim*k + t)
      for t in range(dim):
        if t == l:
          continue
        dL_kk_lt = sum([p[k][t] - p[j][t] for j in neighbors_k])
        val_kl += dL_kk_lt * x[dim*k + l] * x[dim*k + t]

      # ==========================
      # 結果を格納
      # ==========================
      ascend_vec[k, l] = val_kl

  return ascend_vec

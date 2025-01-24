
# G:k-regular graph, p:realization, eps: 近似する際のeps まだ自分で書いた近似関数を使って非ゼロ固有値を求めていた時の関数
def max_p_eigenvalue(G_regular, p, eigen_vec_0, eps, visual_eigen=False, visual_frame=False):
  # 各定数
  # 最大繰り返し回数
  MAX_ITER = 100
  # 移動分の係数
  alpha = 1
  # realization,固有値の格納（格納する固有値は初回を含めて100回分）
  p_box = []
  eigen_vals = []
  eigen_vecs = []
  # 次元 dim
  dim = p.shape[1]
  # node数
  V = len(G_regular)
  # 辺集合
  bonds = np.array(list(G_regular.edges()))
  # 固有ベクトルの初期化
  eigen_vec = eigen_vec_0
  # pを固有ベクトル方向に移動させることで最小非ゼロ固有値の最大化を狙う
  for i in range(MAX_ITER):
    # realizationの格納
    p_box.append(p)
    alpha = pseudo_armijo(alpha, dim, p, bonds)
    print("alpha:",alpha)
    # framework
    F = rp.framework(p, bonds)
    if visual_frame:
      custom_visualize(F)
    # Rigidity matrix
    R = F.rigidityMatrix().T
    # Stiffness matrix
    L = R @ R.T
    # Approximate stiffness matrix
    L_approx = L + eps*np.eye(dim*V)
    eigen_val, eigen_vec = non_zero_eigenvalue(L_approx, eigen_vec)
    # realizationの更新
    p -= alpha*np.copy(eigen_vec).reshape(-1,dim)
    # 固有値の格納
    eigen_vals.append(eigen_val)
    eigen_vecs.append(eigen_vec)
    # フレームワークの削除
    del F
  # 最大値を取るインデックス
  max_index = np.argmax(eigen_vals)
  # visual = Trueの場合に固有値の推移の様子をプロットする。
  if visual_eigen:
    plot_eigen_vals(eigen_vals)
  return p_box[max_index], eigen_vals[max_index], eigen_vecs[max_index]
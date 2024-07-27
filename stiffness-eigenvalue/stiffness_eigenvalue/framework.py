import rigidpy as rp

# (input) realization:p, joints and bars:bonds
def stiffness_matrix(p, bonds):
  F = rp.framework(p,bonds)
  R = F.rigidityMatrix().T
  L = R @ R.T
  del F
  return L
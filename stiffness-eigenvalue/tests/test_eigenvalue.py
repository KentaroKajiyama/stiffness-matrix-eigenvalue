import numpy as np
from stiffness_eigenvalue.eigenvalue import gen_skew_symmetric, gen_parallel_vector, gen_basis

# Sとtの生成のテスト
def test1():
  dim = 2
  S_box = gen_skew_symmetric(dim)
  t_box = gen_parallel_vector(dim)
  for i, trans in enumerate(zip(S_box,t_box)):
    S = trans[0]; t = trans[1]
    print(f"{i}th transformer: S = {S}, t = {t}")
# 基底生成のテスト
def test2():
  dim = 2 
  V = 2
  # position of sites
  p = 5*np.random.randn(dim*V).reshape(-1,dim)
  basis_box = gen_basis(dim,p)
  for i, basis in enumerate(basis_box):
    print(f"{i}th basis: {basis}")
  

if __name__ == "__main__":
  test2()
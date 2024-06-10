from ..stiffness_eigenvalue.min_non_zero_eigenvalue import non_zero_eigenvalue
import numpy as np

def main():
  L = 10*np.random.randn((6,6))
  v0 = np.ones(6)/6
  eps = 1
  eigen_val, eigen_vec = non_zero_eigenvalue(L+eps*np.eye(6), v0)

if __name__ == "__main__":
  main()
import sys
from dotenv import load_dotenv
import os

load_dotenv("~/experiment0124/stiffness-eigenvalue/config/.env")
print(sys.path)
MAX_ITER_FOR_ARMIJO = int(os.getenv("MAX_ITER_FOR_ARMIJO"))

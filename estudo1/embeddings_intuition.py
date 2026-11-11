import numpy as np
from numpy.linalg import norm

vocabulario = {
    "servidor": np.array([0.9, 0.0]),   # Alta tecnologia, zero direito
    "juiz": np.array([0.0, 0.9]),       # Zero tecnologia, alto direito
    "tjam": np.array([0.3, 0.9]),       # Um pouco de tecnologia, muito direito
    "algoritmo": np.array([0.8, 0.0]),  # Alta tecnologia
    "peticao": np.array([0.0, 0.8])     # Alto direito
}
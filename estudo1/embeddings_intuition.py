import numpy as np
from numpy.linalg import norm

vocabulario = {
    "servidor": np.array([0.9, 0.0]),   # Alta tecnologia, zero direito
    "juiz": np.array([0.0, 0.9]),       # Zero tecnologia, alto direito
    "tjam": np.array([0.3, 0.9]),       # Um pouco de tecnologia, muito direito
    "algoritmo": np.array([0.8, 0.0]),  # Alta tecnologia
    "peticao": np.array([0.0, 0.8]),     # Alto direito
    "legaltech": np.array([0.5, 0.5])
}


def similaridade_cosseno(vetor_a, vetor_b):
    return np.dot(vetor_a, vetor_b) / (norm(vetor_a) * norm(vetor_b))

print("=== Comparando Conceitos ===")
sim_servidor_algoritmo = similaridade_cosseno(vocabulario["servidor"], vocabulario["algoritmo"])
print(f"Servidor vs Algoritmo: {sim_servidor_algoritmo:.4f}")

sim_juiz_peticao = similaridade_cosseno(vocabulario["juiz"], vocabulario["peticao"])
print(f"Juiz vs Petição: {sim_juiz_peticao:.4f}")

sim_servidor_juiz = similaridade_cosseno(vocabulario["servidor"], vocabulario["juiz"])
print(f"Servidor vs Juiz: {sim_servidor_juiz:.4f}")

sim_novo = similaridade_cosseno(vocabulario["legaltech"], vocabulario["tjam"])
print(f"legaltech vs tjam: {sim_novo:.4f}")
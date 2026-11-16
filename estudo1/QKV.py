import numpy as np

# Função Softmax corrigida: transforma notas brutas em probabilidades (somam 1.0 ou 100%)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print("=== Simulando Self-Attention com Softmax ===")

# Dimensões: [Tecnologia, Direito]
vocabulario = {
    "o":        np.array([0.05, 0.05]),  # Artigo neutro (pouco peso semântico)
    "servidor": np.array([0.50, 0.50]),  # Palavra ambígua (50% Tech, 50% Direito)
    "na":       np.array([0.05, 0.05]),  # Preposição neutra
    "nuvem":    np.array([0.90, 0.00]),  # Forte Tecnologia
    "publico":  np.array([0.00, 0.90]),  # Forte Direito
    "caiu":     np.array([0.20, 0.10])   # Verbo de contexto leve
}

def processar_atencao(frase, palavra_alvo="servidor"):
    print(f"\n--- Analisando a frase: '{' '.join(frase)}' ---")
    
    # 1. Query (Q): A palavra que está buscando contexto
    q_alvo = vocabulario[palavra_alvo]
    
    # 2. Keys (K) e Values (V): Todas as palavras da frase (incluindo ela mesma)
    keys = [vocabulario[p] for p in frase]
    values = [vocabulario[p] for p in frase]
    
    # 3. Calcula o score bruto (afinidade) entre a Query e cada Key
    scores_brutos = np.array([np.dot(q_alvo, k) for k in keys])
    
    # 4. Aplica o Softmax para transformar scores em pesos percentuais
    pesos_atencao = softmax(scores_brutos)
    
    print("Distribuição de atenção (Softmax):")
    for palavra, score, peso in zip(frase, scores_brutos, pesos_atencao):
        print(f"  - '{palavra:<8}': score bruto = {score:.2f}  -->  atenção = {peso * 100:5.1f}%")
    
    # 5. Soma ponderada dos Values: monta o novo vetor contextualizado
    novo_vetor = np.zeros(2)
    for peso, v in zip(pesos_atencao, values):
        novo_vetor += peso * v
        
    print(f"\nVetor original de '{palavra_alvo}': {q_alvo}")
    print(f"Novo vetor contextualizado : [{novo_vetor[0]:.3f}, {novo_vetor[1]:.3f}]")
    
    if novo_vetor[0] > novo_vetor[1]:
        print("Conclusão da IA: Contexto de TECNOLOGIA (Infraestrutura/TI).")
    else:
        print("Conclusão da IA: Contexto de DIREITO (Funcionalismo Público).")

# Testando os dois cenários com frases completas
processar_atencao(["o", "servidor", "na", "nuvem", "caiu"])
processar_atencao(["o", "servidor", "publico", "assinou"])
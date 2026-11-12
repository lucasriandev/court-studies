from sklearn.tree import DecisionTreeClassifier

# 1. Os dados como vieram da tabela (Humanos)
tamanho_peticao = [[5], [20], [8], [25]] # Características (X)
categorias_texto = ["Cível", "Penal", "Cível", "Penal"] # Respostas (Y)

# 2. O Dicionário de Ida: Textos ➔ Números
tradutor_ia = {
    "Cível": 0,
    "Penal": 1
}

# Usamos o dicionário para converter a lista de textos em lista de números
categorias_numeros = [tradutor_ia[texto] for texto in categorias_texto]
# Resultado: [0, 1, 0, 1]

# 3. Treinando o modelo (O Cérebro entra em ação)
modelo = DecisionTreeClassifier()
# O Scikit-Learn SÓ ACEITA números, por isso passamos a lista convertida
modelo.fit(tamanho_peticao, categorias_numeros) 

# --- AGORA VEM O MUNDO REAL ---

# 4. Chegou um processo novo hoje no Tribunal (ex: 22 páginas)
novo_processo = [[22]]
previsao_ia = modelo.predict(novo_processo)

# A IA não devolve texto, ela devolve um número! (ex: [1])
print(previsao_ia[0]) # Saída: 1

# 5. O Dicionário de Volta: Números ➔ Textos
# (Você só inverte as chaves e os valores)
tradutor_humano = {
    0: "Cível",
    1: "Penal"
}

# Entregando o resultado legível para o juiz ou servidor do TJAM
resultado_final = tradutor_humano[previsao_ia[0]]
print(f"A IA classificou este novo processo como: {resultado_final}") 
# Saída: A IA classificou este novo processo como: Penal
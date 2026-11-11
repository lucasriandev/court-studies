import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

texto = "O tribunal de Justiça do Amazonas julga processsos complexos."

tokens = enc.encode(texto)
print(f"Total de tokens: {len(tokens)}")
print(f"Array de tokens (IDs): {tokens}")

print("\nMapeamento exato:")
for token_id in tokens:
    pedaco = enc.decode([token_id])
    print(f"Token {token_id}: '{pedaco}'")
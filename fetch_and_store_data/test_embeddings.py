from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

try:
    print("Loading model...")
    embedding = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-en",
        model_kwargs={"device": "cpu", "trust_remote_code": True}, #fix loading issue, initialized with random weights getting garbage results
        encode_kwargs={"device": "cpu", "normalize_embeddings": True}
    )
    
    query = "deep branch of radial nerve"
    print(f"Embedding query: '{query}'")
    vec = embedding.embed_query(query)
    
    print(f"Vector length: {len(vec)}")
    print(f"First 5 elements: {vec[:5]}")
    print(f"Norm: {np.linalg.norm(vec)}")
    
    # Test Similarity
    vec2 = embedding.embed_query("radial nerve deep branch")
    similarity = np.dot(vec, vec2)
    print(f"Self-similarity (approx): {similarity:.4f}")
    
    vec3 = embedding.embed_query("banana split ice cream")
    similarity_diff = np.dot(vec, vec3)
    print(f"Dissimilar text similarity: {similarity_diff:.4f}")

    if similarity > 0.8 and similarity_diff < 0.7:
        print("PASS: Model seems to discriminate reasonably.")
    else:
        print("FAIL: Model might be producing random nonsense.")

except Exception as e:
    print(f"Error: {e}")

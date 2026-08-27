from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
chunks = pipeline.retrieve("What is a vector clock, and who proposed it?")

for i, c in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} (score: {c['score']:.3f}, source: {c['source']}) ---")
    print(c['text'][:400])
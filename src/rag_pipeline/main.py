from rag_pipeline.query import query_documents

question = input("Ask a question about your PDF: ")

results = query_documents(question, top_k=3)

for document, metadata, distance in zip(results["documents"][0],
                                        results["metadatas"][0],
                                        results["distances"][0]):
    print(f"\nResult: {document[:500]}")
    print(f"Metadata: {metadata}")
    print(f"Distance: {distance:.4f}")

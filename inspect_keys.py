from rag_engine import load_vectorstore, build_qa_chain

vs = load_vectorstore()
qa_chain, _ = build_qa_chain(vs)

print("Input keys:", qa_chain.input_keys)
print("Output keys:", qa_chain.output_keys)
print("Inner chain type:", type(qa_chain.combine_documents_chain))
print("Inner input vars:", qa_chain.combine_documents_chain.input_keys)

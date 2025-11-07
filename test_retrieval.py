from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("database/faiss_db_FULL", emb, allow_dangerous_deserialization=True)

query = "Explain the procedure for arrest without a warrant under the CrPC."

docs = db.similarity_search(query, k=5)

for i, d in enumerate(docs, 1):
    print(f"\n------ Result {i} ------")
    print(d.page_content[:700])
    print("Source:", d.metadata.get("source", "Unknown"))

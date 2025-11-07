from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import load_vectorstore, build_qa_chain, preprocess_query

app = Flask(__name__)
CORS(app)

print("🔹 Loading FAISS vectorstore and initializing RAG pipeline...")
vector_store = load_vectorstore()
qa_chain, log_top_matches = build_qa_chain(vector_store)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend running successfully."})


@app.route("/query", methods=["POST"])
def query():
    try:
        # --- Read and preprocess the query ---
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("query", "") or data.get("question", "")
        processed_query = preprocess_query(user_query)

        print(f"\n🧠 User Query: {user_query}")
        print(f"🔍 Processed Query: {processed_query}")

        # --- Retrieve top matching docs for transparency ---
        log_top_matches(processed_query)

        # --- Universal model call ---
        print("🔧 Running RAG pipeline...")

        try:
            # 1️⃣ try the common case (query)
            response = qa_chain.invoke({"query": processed_query})
        except Exception as e1:
            try:
                # 2️⃣ fallback (question)
                response = qa_chain.invoke({"question": processed_query})
            except Exception as e2:
                # 3️⃣ last-resort manual call
                print("⚠️ Using manual fallback LLM call")
                docs = qa_chain.retriever.get_relevant_documents(processed_query)
                context = "\n\n".join([d.page_content for d in docs[:5]])
                prompt = f"Question: {processed_query}\n\nContext:\n{context}"
                answer_text = qa_chain.combine_documents_chain.llm_chain.llm.invoke(prompt)
                response = {"result": str(answer_text), "source_documents": docs}

        # --- Extract results ---
        answer = response.get("result", "No answer generated.")
        source_docs = response.get("source_documents", [])

        sources = []
        for doc in source_docs:
            meta = doc.metadata or {}
            doc_type = meta.get("type", "unknown")
            title = meta.get("section_heading", meta.get("case_title", "N/A"))
            preview = doc.page_content[:200].replace("\n", " ")

            sources.append({
                "type": doc_type,
                "file": meta.get("source", "unknown"),
                "title": title,
                "preview": preview
            })

        return jsonify({
            "query": user_query,
            "answer": answer.strip(),
            "sources": sources
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

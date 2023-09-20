from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
import json
from typing import List

class vs(FAISS):
    def do_add(self, paths: List[str], embeddings):
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["answer"] = record.get("answer")
            return metadata
        for path in paths:
            loader = JSONLoader(
                file_path=path,
                jq_schema='.multi_turn_qa[]',
                content_key="question",
                json_lines=True,
                metadata_func=metadata_func
            )
            docs = loader.load()
            new_db = FAISS.from_documents(docs, embeddings)
            db = FAISS.load_local("faiss_index", embeddings)
            db.merge_from(new_db)
            db.save_local("faiss_index")
    def do_search(db, query, topk=1):
        docs = db.similarity_search(query, k=topk)
        qas = []
        for doc in docs:
            qas.append({"question": doc.page_content, "answer": doc.metadata["answer"]})
        return qas
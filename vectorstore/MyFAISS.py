
from typing import Any, Callable, List, Dict
import numpy as np
import copy
import os
import faiss

class MyFAISS():
    def __init__(
            self,
            # embedding_function: Callable,
            # docstore: Docstore,
            # index_to_docstore_id: Dict[int, str],
            normalize_L2: bool = False,
            vs_path: str = "index_file.index",
            dim: int = 768
    ):
        self.vs_path = vs_path
        if os.path.exists(vs_path):
            self.index = faiss.read_index(vs_path) #读入index_file.index文件
        else:
            self.index = faiss.IndexFlatL2(dim)
        self._normalize_L2 = normalize_L2

    def add_vector_doc(self, vectors, doc):
        try:
            for vector in vectors:
                if self._normalize_L2:
                    faiss.normalize_L2(vector)
                self.index.add(vector)
                faiss.write_index(self.index, self.vs_path) #将index保存
                # TODO
                # 将doc与vectors映射起来
        except Exception as e:
            print(e)
            return f"vectors add fail"
    
    def delete_vectors_doc(self, vectors):
        try:
            # TODO
            # 删除index中vectors，删除对应content
            return f"vectors_docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"
    
    def similarity_search_with_score_by_vector(self, vector, topk: int = 10):
        vector = np.array([vector], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        
        scores, results = self.index.search(vector, topk) # 距离和结果

        # TODO
        # 将results对应的content返回
        return results[0]

    
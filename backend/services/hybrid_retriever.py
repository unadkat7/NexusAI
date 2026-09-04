import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config import settings

class HybridRetriever:
    def __init__(self, persistent_path: str = settings.CHROMA_DB_DIR):
        """Initialize ChromaDB vector store and BM25 memory index."""
        os.makedirs(persistent_path, exist_ok=True)
        
        # 1. Setup ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=persistent_path)
        
        # Use default free lightweight embedding model (all-MiniLM-L6-v2)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="nexus_documents",
            embedding_function=self.embedding_fn
        )
        
        # 2. Setup BM25 structures
        self.all_documents: list[Document] = []
        self.bm25 = None
        
        # Reload existing documents from Chroma into BM25 on startup if any exist
        self._reload_bm25_index()

    def _reload_bm25_index(self):
        """Reload stored chunks from ChromaDB into the BM25 keyword index."""
        results = self.collection.get(include=["documents", "metadatas"])
        if results and results["documents"]:
            self.all_documents = []
            tokenized_corpus = []
            
            for doc_content, meta in zip(results["documents"], results["metadatas"]):
                doc = Document(page_content=doc_content, metadata=meta)
                self.all_documents.append(doc)
                tokenized_corpus.append(doc_content.lower().split())
                
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)

    def add_documents(self, documents: list[Document]):
        """
        Ingests document chunks into BOTH ChromaDB (Dense Vector) 
        and BM25 (Sparse Keyword) search indexes.
        """
        if not documents:
            return
            
        ids = []
        contents = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            # Unique doc ID combining filename, chunk_id and existing count
            doc_id = f"{doc.metadata.get('source', 'doc')}_{doc.metadata.get('chunk_id', idx)}_{len(self.all_documents) + idx}"
            ids.append(doc_id)
            contents.append(doc.page_content)
            metadatas.append(doc.metadata)
            
        # Add to ChromaDB Vector Store
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )
        
        # Reload BM25 index to include newly added chunks
        self._reload_bm25_index()
        print(f"✅ Indexed {len(documents)} chunks into ChromaDB & BM25.")

    def dense_search(self, query: str, top_k: int = 10) -> list[Document]:
        """Perform ChromaDB dense semantic vector search."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        retrieved_docs = []
        if results and results["documents"] and results["documents"][0]:
            for doc_content, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_docs.append(Document(page_content=doc_content, metadata=meta))
                
        return retrieved_docs

    def sparse_search(self, query: str, top_k: int = 10) -> list[Document]:
        """Perform BM25 keyword search."""
        if not self.bm25 or not self.all_documents:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair documents with scores, sort descending by score
        doc_scores = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )
        
        # Get top-k non-zero matching documents
        top_docs = []
        for idx, score in doc_scores[:top_k]:
            if score > 0:
                top_docs.append(self.all_documents[idx])
                
        return top_docs

    def hybrid_search(self, query: str, top_k: int = 5, k: int = 60) -> list[Document]:
        """
        Executes Vector Search & BM25 Keyword Search in parallel, 
        then applies RRF (Reciprocal Rank Fusion) to select top-k chunks.
        Formula: RRF_Score(d) = sum(1 / (k + rank))
        """
        vector_results = self.dense_search(query, top_k=top_k * 2)
        bm25_results = self.sparse_search(query, top_k=top_k * 2)
        
        rrf_scores = {}
        doc_map = {}
        
        # Process Vector Search Ranks
        for rank, doc in enumerate(vector_results):
            content_key = doc.page_content.strip()
            doc_map[content_key] = doc
            rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Process BM25 Keyword Ranks
        for rank, doc in enumerate(bm25_results):
            content_key = doc.page_content.strip()
            doc_map[content_key] = doc
            rrf_scores[content_key] = rrf_scores.get(content_key, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Sort documents by combined RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)
        
        # Select top_k chunks
        final_top_k_docs = [doc_map[key] for key in sorted_keys[:top_k]]
        return final_top_k_docs

# Global singleton instance of HybridRetriever
hybrid_retriever = HybridRetriever()

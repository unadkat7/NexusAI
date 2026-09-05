from typing import TypedDict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from config import settings
from services.hybrid_retriever import hybrid_retriever


class RAGState(TypedDict):
    question: str
    chat_history: list[dict]
    top_k: int
    documents: list[Document]
    answer: str
    sources: list[dict]


def _format_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in chat_history:
        role = message.get("role", "message")
        content = message.get("content", "")
        lines.append(f"{role.title()}: {content}")
    return "\n".join(lines)


def _source_from_doc(index: int, doc: Document) -> dict:
    metadata = dict(doc.metadata or {})
    return {
        "id": index,
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "file_path": metadata.get("file_path"),
        "excerpt": doc.page_content[:240],
    }


def _format_context(documents: list[Document]) -> str:
    if not documents:
        return "No retrieved context."

    blocks = []
    for index, doc in enumerate(documents, start=1):
        source = _source_from_doc(index, doc)
        source_label = source["source"]
        if source["page"] is not None:
            source_label = f"{source_label}, page {source['page']}"
        blocks.append(
            f"[{index}] {source_label}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def retrieve(state: RAGState) -> RAGState:
    top_k = max(1, min(int(state.get("top_k", 5)), 10))
    documents = hybrid_retriever.hybrid_search(state["question"], top_k=top_k)
    sources = [_source_from_doc(index, doc) for index, doc in enumerate(documents, start=1)]
    return {**state, "top_k": top_k, "documents": documents, "sources": sources}


def generate(state: RAGState) -> RAGState:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )

    prompt = f"""You are NexusAI, a technical research assistant.
Answer the user's question using the retrieved context first.
Use the conversation history only to resolve references in the question.
If the context does not contain enough information, say what is missing instead of inventing details.
Include concise source markers like [1] or [2] when relying on retrieved context.

Conversation history:
{_format_history(state.get("chat_history", []))}

Retrieved context:
{_format_context(state.get("documents", []))}

Question:
{state["question"]}

Answer:"""

    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))
    return {**state, "answer": answer}


workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_workflow = workflow.compile()

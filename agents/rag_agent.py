"""
RAG Agent — LangGraph-based retrieval-augmented generation agent.
Retrieves relevant context from vector store and generates
grounded responses using OpenAI GPT-4.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class RAGState(TypedDict):
    query: str
    context: List[str]
    response: str


def retrieve_context(state: RAGState) -> RAGState:
    """Retrieve relevant documents from vector store."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("knowledge_base", embeddings)
    docs = vector_store.similarity_search(state["query"], k=5)
    state["context"] = [doc.page_content for doc in docs]
    return state


def generate_response(state: RAGState) -> RAGState:
    """Generate grounded response using retrieved context."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    context_text = "\n\n".join(state["context"])
    prompt = f"""Use the following context to answer the question.
    
Context:
{context_text}

Question: {state["query"]}

Answer based only on the provided context:"""
    response = llm.invoke(prompt)
    state["response"] = response.content
    return state


def build_rag_workflow() -> StateGraph:
    """Build LangGraph RAG workflow."""
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_rag_workflow()
    result = app.invoke({
        "query": "What is the current network status?",
        "context": [],
        "response": ""
    })
    print(result["response"])

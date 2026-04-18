# hr_agent.py - HR Policy Bot using GitHub Models

import os
from datetime import datetime
from typing import TypedDict, List, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load the GITHUB_TOKEN from your .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found. Please make sure you have a .env file with your token.")

# ============================================================
# 1. CONFIGURE GITHUB MODELS (OpenAI-Compatible)
# ============================================================
# The base URL is the inference endpoint for GitHub Models
# We use gpt-4o-mini because it's fast, capable, and within free tier limits
MODEL_NAME = "gpt-4o-mini"
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com/",
    temperature=0.1
)

# ============================================================
# 2. KNOWLEDGE BASE (10 HR policies)
# ============================================================
policies = [
    {"id": "pol_001", "topic": "Work Hours", "text": "Standard work hours are 9 AM to 6 PM, Monday to Friday. Lunch break 1 hour."},
    {"id": "pol_002", "topic": "Leave Policy", "text": "Employees get 20 paid leaves per year. Unused leaves carry over up to 10 days."},
    {"id": "pol_003", "topic": "Sick Leave", "text": "12 paid sick leaves per year. Medical certificate required for 3+ consecutive days."},
    {"id": "pol_004", "topic": "Public Holidays", "text": "Company observes 10 national holidays: Republic Day, Independence Day, Diwali, etc."},
    {"id": "pol_005", "topic": "Salary Day", "text": "Salary is credited on the last working day of every month."},
    {"id": "pol_006", "topic": "WFH Policy", "text": "Work from home allowed 2 days per week with manager approval."},
    {"id": "pol_007", "topic": "Performance Bonus", "text": "Annual bonus up to 15% of salary based on performance review."},
    {"id": "pol_008", "topic": "Notice Period", "text": "Notice period is 30 days for resignation. Can be waived by HR."},
    {"id": "pol_009", "topic": "Health Insurance", "text": "Company provides health insurance up to ₹5 lakhs for employee and family."},
    {"id": "pol_010", "topic": "HR Helpline", "text": "For any unclear policy, call HR at 1800-123-4567 or email hr@company.com."},
]

# ============================================================
# 3. BUILD CHROMADB VECTOR STORE
# ============================================================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("hr_policies")

for doc in policies:
    emb = embedder.encode(doc["text"]).tolist()
    collection.add(
        ids=[doc["id"]],
        embeddings=[emb],
        documents=[doc["text"]],
        metadatas=[{"topic": doc["topic"]}]
    )

print("✅ ChromaDB ready")

# ============================================================
# 4. STATE DEFINITION
# ============================================================
class HRState(TypedDict):
    question: str
    messages: Annotated[List, add_messages]
    route: Literal["retrieve", "tool", "skip"]
    retrieved: str
    sources: List[dict]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str

def create_initial_state(question: str) -> HRState:
    return {
        "question": question,
        "messages": [],
        "route": "retrieve",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name": "",
    }

# ============================================================
# 5. NODE FUNCTIONS
# ============================================================
def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Helper function to call GitHub Models via LangChain"""
    response = llm.invoke(prompt)
    return response.content

def memory_node(state: HRState):
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": state["question"]})
    if len(messages) > 6:
        messages = messages[-6:]
    user_name = state.get("user_name", "")
    if "my name is" in state["question"].lower():
        parts = state["question"].lower().split("my name is")
        if len(parts) > 1:
            user_name = parts[1].strip().split()[0].capitalize()
    return {"messages": messages, "user_name": user_name}

def router_node(state: HRState):
    prompt = f"""You are a router for an HR policy bot. Decide:
- 'retrieve' if the question asks about company policies, leaves, salary, holidays, insurance, etc.
- 'tool' if it asks for current date, time, or calculation.
- 'skip' if it's a greeting, thank you, or memory-only (e.g., "what's my name?").
Question: {state['question']}
Answer only one word: retrieve, tool, or skip."""
    route = call_llm(prompt, temperature=0.0).strip().lower()
    if route not in ["retrieve", "tool", "skip"]:
        route = "retrieve"
    return {"route": route}

def retrieval_node(state: HRState):
    emb = embedder.encode(state["question"]).tolist()
    results = collection.query(emb, n_results=3)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    context = "\n\n".join([f"[{m['topic']}] {d}" for d, m in zip(docs, metas)])
    return {"retrieved": context, "sources": metas}

def skip_retrieval_node(state: HRState):
    return {"retrieved": "", "sources": []}

def tool_node(state: HRState):
    q = state["question"].lower()
    if "date" in q or "today" in q:
        return {"tool_result": f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"}
    elif "time" in q:
        return {"tool_result": f"The current time is {datetime.now().strftime('%I:%M %p')}"}
    else:
        return {"tool_result": "Tool could not answer. Please ask HR directly."}

def answer_node(state: HRState):
    system = """You are an HR policy assistant. Answer ONLY using the provided context.
If the context does NOT contain the answer, say: "I don't know. Please call HR helpline 1800-123-4567."
Never invent policies or numbers. Be helpful but concise."""
    
    if state.get("retrieved"):
        context = state["retrieved"]
    elif state.get("tool_result"):
        context = f"Tool result: {state['tool_result']}"
    else:
        context = "No context provided."
    
    user_name = state.get("user_name", "")
    greeting = f"User name is {user_name}. " if user_name else ""
    
    full_prompt = f"{system}\n\n{greeting}Context:\n{context}\n\nQuestion: {state['question']}\nAnswer:"
    answer = call_llm(full_prompt, temperature=0.1)
    return {"answer": answer}

def eval_node(state: HRState):
    if not state.get("retrieved"):
        return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0) + 1}
    
    prompt = f"""Rate faithfulness from 0.0 to 1.0, where 1.0 means the answer uses ONLY information from the context.
Context: {state['retrieved']}
Answer: {state['answer']}
Score only (a number between 0 and 1):"""
    try:
        score = float(call_llm(prompt, temperature=0.0).strip())
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": state.get("eval_retries", 0) + 1}

def save_node(state: HRState):
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state["answer"]})
    return {"messages": messages}

# ============================================================
# 6. GRAPH ASSEMBLY
# ============================================================
def route_decision(state: HRState) -> Literal["retrieve", "skip", "tool"]:
    return state["route"]

def eval_decision(state: HRState) -> Literal["answer", "save"]:
    if state["faithfulness"] >= 0.7 or state["eval_retries"] >= 2:
        return "save"
    else:
        return "answer"

graph = StateGraph(HRState)
graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")
graph.add_edge("memory", "router")
graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())
print("✅ LangGraph compiled successfully with GitHub Models.\n")

# ============================================================
# 7. HELPER FUNCTION
# ============================================================
def ask(question: str, thread_id: str = "default_user") -> str:
    initial = create_initial_state(question)
    result = app.invoke(
        initial,
        config={"configurable": {"thread_id": thread_id}}
    )
    return result["answer"]

if __name__ == "__main__":
    print(ask("How many paid leaves do I get?"))
    print(ask("My name is John", "user1"))
    print(ask("What is my name?", "user1"))
    print(ask("What is the capital of France?"))
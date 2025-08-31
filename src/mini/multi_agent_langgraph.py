# Planner, Retriever, Summarizer, Tools, Memory, Graph build
# pip install langgraph langchain openai faiss-cpu duckduckgo-search
from __future__ import annotations

import os
from typing import Dict, Any
# LangGraph
from langgraph.graph import StateGraph, END
# LangChain
from langchain.chat_models import ChatOpenAI, ChatOllama # site-packages\langchain\chat_models\__init__.py
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
# Prompts & Chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Conversation memory
from langchain.memory import ConversationBufferMemory

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # "llama3.2" "mistral"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Create shared memory
memory = ConversationBufferMemory(return_messages=True)

def get_history_text() -> str:
    # Flatten message history into a single string for prompts
    # memory.buffer is already a concatenated string in many LangChain versions.
    # If not, join from memory.chat_memory.messages.
    try:
        return memory.buffer
    except Exception:
        msgs = getattr(memory, "chat_memory", None)
        if msgs and getattr(msgs, "messages", None):
            return "\n".join(f"{m.type.upper()}: {m.content}" for m in msgs.messages)
        return ""

def remember(role: str, content: str) -> None:
    # Append a message to memory
    if role.lower() == "user":
        memory.chat_memory.add_user_message(content)
    else:
        memory.chat_memory.add_ai_message(content)

# Tools (Retriever uses search)
# tools = [
#     Tool(
#         name="DuckDuckGo Search",
#         func=DuckDuckGoSearchRun().run,
#         description="Search the web using DuckDuckGo"
#     )
# ]
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Useful for answering general knowledge queries and recent events."
)
tools = [search_tool] 
# tools.append(search_tool)

# Local LLM via Ollama
llm = ChatOllama(model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE)

# Define the LLM for Each Agent
# llm = ChatOpenAI(model="gpt-4", temperature=0) # temperature from 0 to 1 that controls randomness for example is: 0.5 that makes the output more random and creative, 1 that makes it more focused and deterministic

# Prompts & Chains
planner_prompt = PromptTemplate.from_template(
    "You are a planner agent.\n"
    "Conversation history:\n{history}\n\n"
    "Break down the user task into a short, numbered list of actionable steps.\n"
    "Task: {input}\n"
)
planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

retriever_prompt = PromptTemplate.from_template(
    "You are a retrieval agent.\n"
    "Conversation history:\n{history}\n\n"
    "Use web search to find information that helps execute these steps:\n{planner_output}\n"
    "Return a concise set of findings (bullet points, include sources if available)."
)
retriever_chain = LLMChain(llm=llm, prompt=retriever_prompt)

summarizer_prompt = PromptTemplate.from_template(
    "You are a summarizer agent.\n"
    "Conversation history:\n{history}\n\n"
    "Summarize the following retrieved information for the user in a clear, actionable way:\n"
    "{retrieved_info}\n"
    "Keep it concise, factual, and well-structured."
)
summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt)

# Node Functions (LangGraph)
def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:  state['input'] - user request
    Output: planner_output - step list
            step           - next node id
    """
    user_input = state["input"]
    remember("user", user_input)

    steps = planner_chain.run({"input": user_input, "history": get_history_text()})
    remember("ai", f"[Planner]\n{steps}")

    return {"planner_output": steps, "step": "retrieve"}

def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:  planner_output - what to search for
    Output: retrieved_info - raw findings (may include URLs/snippets)
            step           - next node id
    """
    query_plan = state["planner_output"]
    # Option A (tool-less): let LLM compose a query & summarize results itself.
    # Option B (hybrid): actually call the search tool to fetch snippets, then ask LLM to organize.
    
    # From Option A& B => fetch with tool, then ask LLM to organize + add context.

    # tool call
    tool_results = search.run(query_plan)

    # LLm format n enrich
    retrieved_info = retriever_chain.run(
        {"planner_output": query_plan, "history": get_history_text()}
    )

    # Merge tool raw result into the retrieved_info for more grounding
    merged_info = f"{retrieved_info}\n\n[Raw search result]\n{tool_results[:2000]}"

    remember("ai", f"[Retriever]\n{merged_info}")
    return {"retrieved_info": merged_info, "step": "summarize"}

def summarizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:  retrieved_info - evidence
    Output: final_output   - user-facing summary
            step           - END
    """
    final_output = summarizer_chain.run(
        {"retrieved_info": state["retrieved_info"], "history": get_history_text()}
    )
    remember("ai", f"[Summarizer]\n{final_output}")
    return {"final_output": final_output, "step": END}

# Build the Graph
def build_app():
    graph = StateGraph(dict)  # dict-based state
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("summarize", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

# Run the Multi-Agent System
if __name__ == "__main__":
    app = build_app()
    # Example query (recent topic to exercise web search)
    user_request = "Tell me what's new about Mars exploration this month."
    result = app.invoke({"input": user_request})
    print("\n=== Final Output ===\n")
    print(result.get("final_output", "No output"))
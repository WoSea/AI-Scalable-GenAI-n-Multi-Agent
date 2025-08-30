# pip install langgraph langchain openai faiss-cpu duckduckgo-search
from langgraph.graph import StateGraph
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create shared memory
memory = ConversationBufferMemory(return_messages=True)

# Define tools for the retrieval agent
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=DuckDuckGoSearchRun(),
        description="Search the web using DuckDuckGo"
    )
]

search = DuckDuckGoSearchRun()
search_tool = Tool(
name="WebSearch",func=search.run,
description="Useful for answering general knowledge queries"
)

# Add the search tool to the tools list
tools.append(search_tool)

# Define the LLM for Each Agent
llm = ChatOpenAI(model="gpt-4", temperature=0) # temperature from 0 to 1 that controls randomness for example is: 0.5 that makes the output more random and creative, 1 that makes it more focused and deterministic

# Define Agent Prompts and Functions
planner_prompt = PromptTemplate.from_template(
    "You are a planner agent. Break down the task: {input}into a list of steps."
)
planner_chain = LLMChain(llm=llm, prompt=planner_prompt)
def planner_node(state):
    steps = planner_chain.run(state["input"])
    return {"planner_output": steps, "step": "retrieve"}

# Retrieve Node
retriever_prompt = PromptTemplate.from_template(
    "Use the web search to find info on: {planner_output}"
) 
retriever_chain = LLMChain(llm=llm, prompt=retriever_prompt)
def retriever_node(state):
    query = state["planner_output"]
    result = search.run(query)
    return {"retrieved_info": result, "step": "summarize"} # return {"search_results": results, "step": "synthesize"}

# Summarizer Agent
summarizer_prompt = PromptTemplate.from_template(
"Summarize the following information for the user: {retrieved_info}"
) 
summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt)

def summarizer_node(state):
    final_output = summarizer_chain.run(state["retrieved_info"])
    return {"final_output": final_output, "step": "end"}

# Build the LangGraph
graph = StateGraph()
graph.add_node("planner", planner_node)
graph.add_node("retrieve", retriever_node)
graph.add_node("summarize", summarizer_node)
graph.set_entry_point("planner")
graph.add_edge("planner", "retrieve")
graph.add_edge("retrieve", "summarize")
graph.add_edge("summarize", "end")
app = graph.compile()

# Run the Multi-Agent System
response = app.invoke({"input": "Tell me what's new about Mars exploration this month."})
print("\nFinal Output:\n", response["final_output"])
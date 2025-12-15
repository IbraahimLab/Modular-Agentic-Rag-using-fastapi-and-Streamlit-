from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from server.agent.state import AgentState

def build_agent(model_name: str, groq_api_key: str, tools):
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        groq_api_key=groq_api_key,
    ).bind_tools(tools)


    SYSTEM_PROMPT = """
You are an agentic RAG assistant.

Rules:
- If a PDF search tool is available, you MUST use it for questions
  that can be answered from the uploaded PDF.
- Use web search only if the PDF does not contain the answer.
- Never claim you do not have a PDF if a PDF tool exists.
- Use arxiv search only if the PDF does not contain the answer.
- If you are not sure how to answer a question, you can use web search
  to find the answer.
"""

    def llm_node(state: AgentState):
        messages = state["messages"]

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm.invoke(messages)
        return {"messages": [response]} 

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(tools))  


    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "llm")

    return graph.compile()

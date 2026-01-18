from typing import Literal, Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ────────────────────────────────────────────────
#   Router Decision Schema
# ────────────────────────────────────────────────
class RouteDecision(BaseModel):
    """Decide which specialized agent should handle the query"""
    next_agent: Literal["sql_agent", "wiki_agent", "clarify"] = Field(
        description="Which agent should process this query"
    )
    reasoning: str = Field(description="Short explanation why you chose this agent")


# ────────────────────────────────────────────────
#   State
# ────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    final_answer: str | None
    decision: RouteDecision | None


# ────────────────────────────────────────────────
#   LLM & Router setup
# ────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent router between two specialized agents:

1. sql_agent     → database questions, SQL generation, filtering data, aggregations,
                  counts, trends, business metrics, "how many", "what is the total",
                  "top 10", "average", "group by", any question that needs database

2. wiki_agent    → general knowledge, explanations, definitions, history,
                  science, geography, biographies, "what is", "who is", "how does",
                  comparisons, concepts, trivia

3. clarify       → question is too vague / ambiguous / doesn't clearly belong to above

Choose very carefully — most business/reporting/analytics questions should go to sql_agent."""),
    ("placeholder", "{messages}"),
])

router_chain = router_prompt | llm.with_structured_output(RouteDecision)


# ────────────────────────────────────────────────
#   Router Node
# ────────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    decision: RouteDecision = router_chain.invoke(state["messages"])

    # Optional: show decision in conversation (good for debugging)
    state["messages"].append(AIMessage(content=f"Router → {decision.next_agent}\nReason: {decision.reasoning}"))

    return {
        "decision": decision,
        "messages": state["messages"]
    }


# ────────────────────────────────────────────────
#   Agent Wrappers
#   (replace with your actual agent executors)
# ────────────────────────────────────────────────
def sql_agent_node(state: AgentState) -> dict:
    # Your real SQL agent here
    # agent_executor = create_SQL_agent(llm, tools)

    message = state["messages"][0].content  # original user question

    # result = agent_executor.invoke({"input": message})
    # simulated:
    result = {
        "output": f"SQL Agent result for: {message}\n"
                  f"(executed query, returned data, explanation...)"
    }

    return {"final_answer": result["output"]}


def wiki_agent_node(state: AgentState) -> dict:
    # Your real wiki/general agent here
    # agent_executor = create_wiki_agent(llm, tools)

    message = state["messages"][0].content

    # result = agent_executor.invoke({"input": message})
    # simulated:
    result = {
        "output": f"Wiki / Knowledge result for: {message}\n"
                  f"(summary, explanation, facts...)"
    }

    return {"final_answer": result["output"]}


def clarify_node(state: AgentState) -> dict:
    return {
        "final_answer": "I'm not sure whether this question should be answered with SQL/database "
                        "or general knowledge.\n\n"
                        "Could you clarify if you're asking about:\n"
                        "• data from database / reports / metrics\n"
                        "• or general information / explanation / definition?"
    }


# ────────────────────────────────────────────────
#   Routing function
# ────────────────────────────────────────────────
def route_after_router(state: AgentState) -> Literal["sql_agent", "wiki_agent", "clarify"]:
    return state["decision"].next_agent


# ────────────────────────────────────────────────
#   Build Graph
# ────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("wiki_agent", wiki_agent_node)
workflow.add_node("clarify", clarify_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "sql_agent": "sql_agent",
        "wiki_agent": "wiki_agent",
        "clarify": "clarify",
    }
)

workflow.add_edge("sql_agent", END)
workflow.add_edge("wiki_agent", END)
workflow.add_edge("clarify", END)

graph = workflow.compile()


# ────────────────────────────────────────────────
#   Quick usage example
# ────────────────────────────────────────────────
def ask(question: str):
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "final_answer": None,
        "decision": None
    })
    return result["final_answer"]


# Test cases
if __name__ == "__main__":
    print(ask("How many orders were placed last month?"))
    # → sql_agent

    print(ask("What is the capital of Iceland?"))
    # → wiki_agent

    print(ask("Can you help me?"))
    # → clarify

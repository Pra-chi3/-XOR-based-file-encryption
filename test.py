from typing import Literal, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# -------------------------------------------------------------------------
#   Router Decision Schema
# -------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """Decide which agent should handle the current user query"""
    next_agent: Literal["sql_agent", "wiki_agent", "clarify"] = Field(
        description="Which specialized agent should process this query"
    )
    reasoning: str = Field(
        description="Short explanation of the routing decision"
    )
    # Optional parameters - useful especially for sql_agent
    time_period: str = Field(default="", description="Detected time period if mentioned")
    main_topic: str = Field(default="", description="Main subject/entity of the question")


# -------------------------------------------------------------------------
#   Graph State
# -------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    decision: RouteDecision | None
    final_answer: str | None


# -------------------------------------------------------------------------
#   LLM & Router Setup
# -------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a smart router between two specialized agents:

1. sql_agent     → questions that require data from database:
   - metrics, counts, sums, averages
   - filters, time periods, trends
   - "how many", "what is the total", "show me", "list", "top", "last month/year"
   - any question that needs real numbers or records

2. wiki_agent    → general knowledge questions:
   - definitions, explanations, concepts
   - who/what/when/where/why/how
   - history, science, geography, biographies
   - comparisons, trivia, general facts

3. clarify       → ambiguous, too vague, mixed intent, or doesn't clearly fit above

Be conservative: when in doubt → choose 'clarify'
Most business, analytics, reporting questions should go to sql_agent."""),
    ("placeholder", "{messages}"),
])

router_chain = router_prompt | llm.with_structured_output(RouteDecision)


# -------------------------------------------------------------------------
#   Nodes
# -------------------------------------------------------------------------
def router_node(state: AgentState) -> dict:
    """Classify user intent and decide next step"""
    decision: RouteDecision = router_chain.invoke(state["messages"])

    # Optional: add routing decision to conversation (helps debugging)
    routing_msg = AIMessage(
        content=f"[Router] → {decision.next_agent}\nReason: {decision.reasoning}"
    )
    
    return {
        "decision": decision,
        "messages": state["messages"] + [routing_msg]
    }


def sql_agent_node(state: AgentState) -> dict:
    """Execute SQL / Database agent"""
    # In real system → call your actual SQL agent here
    original_question = state["messages"][0].content
    
    # Example placeholder result
    simulated_output = (
        f"SQL Agent executed for: {original_question}\n\n"
        f"→ Generated SQL query\n"
        f"→ Fetched data from database\n"
        f"→ Formatted answer with explanation"
    )
    
    return {"final_answer": simulated_output}


def wiki_agent_node(state: AgentState) -> dict:
    """Execute Wiki / General Knowledge agent"""
    # In real system → call your actual wiki/general agent here
    original_question = state["messages"][0].content
    
    # Example placeholder
    simulated_output = (
        f"Knowledge/Wiki answer for: {original_question}\n\n"
        f"→ Relevant explanation\n"
        f"→ Key facts\n"
        f"→ Sources/context"
    )
    
    return {"final_answer": simulated_output}


def clarify_node(state: AgentState) -> dict:
    """Ask user to clarify intent"""
    clarification_text = (
        "I'm not sure whether your question is about:\n\n"
        "• **Database / numbers / reports / metrics**  (sales, counts, trends, etc.)\n"
        "• **General knowledge / explanation / definition**  (concepts, facts, history)\n\n"
        "Please reply with one of these to help me choose the right tool:\n"
        "• database / data / numbers / report\n"
        "• knowledge / explanation / general / definition\n\n"
        "Or just rephrase your question with more context. Thank you! 😊"
    )
    
    return {"final_answer": clarification_text}


# -------------------------------------------------------------------------
#   Routing Logic
# -------------------------------------------------------------------------
def route_decision(state: AgentState) -> Literal["sql_agent", "wiki_agent", "clarify"]:
    return state["decision"].next_agent


# -------------------------------------------------------------------------
#   Build & Compile Graph
# -------------------------------------------------------------------------
workflow = StateGraph(state_schema=AgentState)

workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("wiki_agent", wiki_agent_node)
workflow.add_node("clarify", clarify_node)

# Flow
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "sql_agent": "sql_agent",
        "wiki_agent": "wiki_agent",
        "clarify": "clarify",
    }
)

# All terminal nodes go to END
workflow.add_edge("sql_agent", END)
workflow.add_edge("wiki_agent", END)
workflow.add_edge("clarify", END)

# Compile the graph
graph = workflow.compile()


# -------------------------------------------------------------------------
#   Helper to run the graph
# -------------------------------------------------------------------------
def ask(question: str):
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "decision": None,
        "final_answer": None
    })
    return result["final_answer"]


# -------------------------------------------------------------------------
#   Quick tests
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Test 1:", ask("How many customers registered in December 2025?"))
    print("\nTest 2:", ask("What is the difference between INNER JOIN and LEFT JOIN?"))
    print("\nTest 3:", ask("Can you help me please?"))

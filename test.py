from typing import Literal, Annotated
from typing_extensions import TypedDict
import json

from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =============================================
#   Router Decision Schema
# =============================================
class RouteDecision(BaseModel):
    next_agent: Literal["sql_agent", "wiki_agent", "clarify"] = Field(
        description="Which agent should handle this query"
    )
    reasoning: str = Field(description="Short explanation of the decision")


# =============================================
#   Graph State
# =============================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    decision: RouteDecision | None
    final_answer: str | None


# =============================================
#   Router prompt (forces JSON output)
# =============================================
ROUTER_SYSTEM_PROMPT = """\
You are a smart router that decides which specialized agent should answer the user.

Available agents:
- sql_agent     → questions about database content, metrics, counts, sums, filters,
                  time periods, trends, business numbers, "how many", "total", "last month", reports
- wiki_agent    → general knowledge, definitions, explanations, history, science,
                  geography, who/what/when/where/why, concepts, trivia
- clarify       → too vague, ambiguous, doesn't clearly fit above categories

Rules:
- Most business/analytics/reporting questions → sql_agent
- Pure explanations/definitions/concepts → wiki_agent
- When unsure → clarify

Respond **ONLY** with valid JSON object in this exact format, nothing else:

{
  "next_agent": "sql_agent" | "wiki_agent" | "clarify",
  "reasoning": "one sentence explanation"
}

Do not add any extra text, comments, markdown or code blocks.
"""


# =============================================
#   Router Node (JSON + fallback)
# =============================================
def router_node(state: AgentState) -> dict:
    # Prepare prompt with current conversation
    user_question = state["messages"][0].content

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"User question: {user_question}"}
    ]

    try:
        raw_response = llm.invoke(messages)  # your custom LLMClient

        # Attempt to find and parse JSON
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found")

        json_str = raw_response[start:end]
        parsed = json.loads(json_str)
        decision = RouteDecision.model_validate(parsed)

    except (json.JSONDecodeError, ValidationError, ValueError, Exception) as e:
        # Fallback when output is malformed
        decision = RouteDecision(
            next_agent="clarify",
            reasoning=f"Router failed to parse structured output: {str(e)}"
        )

    # Optional: add decision to messages (debugging & transparency)
    state["messages"].append(AIMessage(
        content=f"[Routing decision]\n"
                f"Agent: {decision.next_agent}\n"
                f"Reason: {decision.reasoning}"
    ))

    return {
        "decision": decision,
        "messages": state["messages"]
    }


# =============================================
#   Agent nodes (placeholders)
# =============================================
def sql_agent_node(state: AgentState) -> dict:
    original_question = state["messages"][0].content
    # ← Replace with your real sql agent call
    simulated = (
        f"SQL Agent result for: {original_question}\n"
        "(simulated: query executed, data retrieved, formatted answer)"
    )
    return {"final_answer": simulated}


def wiki_agent_node(state: AgentState) -> dict:
    original_question = state["messages"][0].content
    # ← Replace with your real wiki/general agent call
    simulated = (
        f"Knowledge/Wiki answer for: {original_question}\n"
        "(simulated: explanation, facts, summary)"
    )
    return {"final_answer": simulated}


def clarify_node(state: AgentState) -> dict:
    clarification = (
        "I'm not sure whether your question is about:\n\n"
        "• Database / numbers / metrics / reports / business data\n"
        "• General knowledge / explanation / definition / concepts\n\n"
        "Please reply with one of these to help me choose:\n"
        "• database / data / numbers / report\n"
        "• knowledge / explanation / general / definition\n\n"
        "Or rephrase your question with more context. Thank you!"
    )
    return {"final_answer": clarification}


# =============================================
#   Routing function
# =============================================
def route_after_router(state: AgentState) -> str:
    return state["decision"].next_agent


# =============================================
#   Build Graph
# =============================================
workflow = StateGraph(state_schema=AgentState)

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


# =============================================
#   Simple runner
# =============================================
def ask(question: str) -> str:
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "decision": None,
        "final_answer": None
    })
    return result["final_answer"]


# Quick test
if __name__ == "__main__":
    print(ask("How many customers signed up last month?"))
    # print(ask("What is photosynthesis?"))
    # print(ask("Hi how are you?"))



















from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field

import httpx
import json


class CustomGenFactoryChat(BaseChatModel):
    """
    Minimal custom chat model for your internal GenFactory endpoint.
    Adjust payload / headers / parsing as needed.
    """

    # ── Required / strongly recommended fields ──
    base_url: str = Field(..., description="Base URL of the API")
    api_key: str = Field(..., description="API key (if required)")
    model_name: str = Field(..., alias="model")  # allows model=... in constructor

    temperature: float = 0.2
    max_tokens: Optional[int] = 2048

    # You can add your own fields
    default_system: Optional[str] = None   # optional forced system prompt

    # Internal http client (can be passed or created here)
    http_client: Optional[httpx.Client] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.http_client is None:
            self.http_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=15.0, read=90.0),
                # proxies=..., http2=..., etc.
            )

    @property
    def _llm_type(self) -> str:
        return "genfactory-custom-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    def _create_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict:
        # ── Turn LangChain messages → your endpoint format ──
        payload_messages = []

        # Optional: inject fixed system prompt
        if self.default_system:
            payload_messages.append({"role": "system", "content": self.default_system})

        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                role = "user"  # fallback

            content = msg.content
            # You can add .tool_calls handling here if needed

            payload_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if stop:
            payload["stop"] = stop

        # Add any extra fixed fields your endpoint requires
        # payload["top_p"] = 0.95
        # payload["stream"] = False

        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._create_payload(messages, stop=stop, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Add any custom headers your proxy requires
            # "X-Custom-Header": "value",
        }

        try:
            response = self.http_client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            # ── Adapt your endpoint's response shape here ──
            # Most common patterns:
            content = data["choices"][0]["message"]["content"]
            # or: data["choices"][0]["delta"]["content"]  (if mistaken for stream)
            # or: data["response"] / data["text"] / ...

            message = AIMessage(content=content)

            # If your endpoint returns tool_calls / usage / finish_reason:
            # message.tool_calls = ... 
            # generation_info = {"finish_reason": ..., "usage": ...}

            generation = ChatGeneration(message=message)  # , generation_info=...)

            return ChatResult(generations=[generation])

        except Exception as e:
            # You can raise custom errors or log here
            raise RuntimeError(f"GenFactory call failed: {str(e)}") from e

    # Optional: if you want streaming support
    # async def _astream(...) or def _stream(...)
    # ── see full docs for astream_chunks / token-by-token yielding ──

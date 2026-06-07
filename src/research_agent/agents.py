"""ReAct and Chain-of-Thought agents + the corrected router.

Ported from notebook cells 22-23 and 27. Key changes:
  * The router (`answer`) now ALWAYS runs the chosen strategy; specialized tools
    are injected as a hint, never a hard bypass (improvement #8, via routing.py).
  * `max_iterations` default lowered 15 -> 8 to curb ReAct over-searching (#6).
  * find_paper_quotes is not yet ported, so prompts reference only the 6 live
    tools; the CoT executor safely skips any unknown tool name in a plan.
"""

from __future__ import annotations

import json
import re

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from .routing import (
    build_agent_input,
    choose_fallback_tool,
    decide_route,
    needs_tool_fallback,
    normalize_query,
)
from .tools import BASE_TOOLS
from .paper_quotes import find_paper_quotes

DEFAULT_LLM_MODEL = "gpt-4o-mini"

# The agents use all 7 tools (6 base + the find_paper_quotes pipeline).
ALL_TOOLS = BASE_TOOLS + [find_paper_quotes]

REACT_SYSTEM = """You are a Research Assistant Agent. Your job is to answer complex research questions.
1. Break the question into sub-questions if needed.
2. Use tools to gather information - do NOT make up facts.
3. Cite every claim by noting which tool result or URL it came from.
4. Do not invent citation metadata. If author, date, venue, or title is not visible in tool results, say it is not identified or use n.d.
5. Provide a "Sources" section at the end with source titles and URLs whenever URLs are available.
6. For non-trivial research, study-advice, current-information, or factual-explanation questions, use at least one appropriate tool before answering.
7. Only answer without tools for greetings, creative writing, simple arithmetic, or very short conversational replies.
8. Be efficient - avoid redundant searches. Once you have enough information, stop searching and write the answer.
9. For exact verified quotes from the active document use quote_search; to summarize it use summarize_active_document. When the user gives a paragraph or claim and asks for papers, sources, references, or supporting quotes, use find_paper_quotes.
10. Format citations directly in your answer; only call generate_citation for academic papers where precise formatting matters, passing only metadata present in the source."""

COT_SYSTEM = """You are a Research Assistant Agent. You answer complex research questions by planning your research strategy upfront, then executing it.

You have access to these tools:
1. web_search(query) - current information from the web
2. wikipedia_search(query) - stable background context
3. fetch_pdf(url) - download and extract text from a PDF
4. generate_citation(title, authors, year, url, source_type) - format a citation
5. quote_search(query) - exact verified quotes from the currently loaded document
6. summarize_active_document(query) - summarize/answer about the active document
7. find_paper_quotes(query) - find external papers/sources + verified supporting quotes for a paragraph or claim

Choose tools according to the user's intent. For exact quotes or evidence from the active document, include quote_search. For "what does the uploaded document say" / summaries, include summarize_active_document. For non-trivial research, include at least one tool. Use an empty plan only for greetings, creative writing, arithmetic, or short conversational replies.

Output ONLY a JSON research plan, no other text:
```json
{"plan": [{"tool": "tool_name", "input": {"query": "value"}}]}
```"""


def _tool_map(tools):
    return {t.name: t for t in tools}


def run_react(query, tools=None, model_name=DEFAULT_LLM_MODEL, temperature=0.0,
              max_iterations=8, verbose=False):
    """ReAct agent on LangGraph (#5): explicit graph state + recursion-limit
    stopping, replacing the langchain-classic AgentExecutor. Returns the same
    shape as before (output / intermediate_steps / num_steps / tools_used) so the
    router, grounding pass, and tests keep working."""
    tools = tools if tools is not None else ALL_TOOLS
    query = normalize_query(query)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    graph = create_react_agent(llm, tools, prompt=REACT_SYSTEM)
    tool_map = _tool_map(tools)

    output, steps, tools_used = "", [], []
    try:
        state = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 2 * max_iterations + 1},
        )
        messages = state["messages"]
        pending = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    pending[tc["id"]] = tc["name"]
            elif isinstance(msg, ToolMessage):
                name = pending.get(msg.tool_call_id) or getattr(msg, "name", "tool")
                tools_used.append(name)
                steps.append((name, str(msg.content)))
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
    except GraphRecursionError:
        output = "Stopped: reached the maximum number of reasoning steps before finishing."

    # No-tool fallback backstop (parity with the legacy executor version).
    if not tools_used and needs_tool_fallback(query):
        fb_name = choose_fallback_tool(query)
        fb = tool_map.get(fb_name)
        if fb is not None:
            fb_result = fb.invoke(query)
            tools_used = [fb_name]
            steps = [(fb_name, fb_result)]
            synthesis = llm.invoke([
                SystemMessage(content="Revise the draft answer using the tool result below. Ground "
                              "factual claims in it, cite source titles/URLs when available, and do "
                              "not invent citation metadata."),
                HumanMessage(content=f"Question: {query}\n\nDraft answer:\n{output}\n\n"
                             f"Tool used: {fb_name}\nTool result:\n{fb_result}"),
            ])
            output = synthesis.content

    return {"output": output, "intermediate_steps": steps,
            "num_steps": len(steps), "tools_used": tools_used}


def run_react_executor(query, tools=None, model_name=DEFAULT_LLM_MODEL, temperature=0.0,
                       max_iterations=8, verbose=False):
    """Legacy ReAct agent on the langchain-classic AgentExecutor. Kept as a
    fallback / for comparison; run_react now uses LangGraph."""
    tools = tools if tools is not None else ALL_TOOLS
    query = normalize_query(query)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_SYSTEM),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose,
                             max_iterations=max_iterations, return_intermediate_steps=True)
    result = executor.invoke({"input": query})
    steps = result.get("intermediate_steps", [])
    tools_used = [s[0].tool for s in steps]
    output = result["output"]

    if not tools_used and needs_tool_fallback(query):
        fb_name = choose_fallback_tool(query)
        fb = _tool_map(tools).get(fb_name)
        if fb is not None:
            fb_result = fb.invoke(query)
            tools_used = [fb_name]
            steps = [(fb_name, fb_result)]
            synthesis = llm.invoke([
                SystemMessage(content="Revise the draft answer using the tool result below. Ground "
                              "factual claims in it, cite source titles/URLs when available, and do "
                              "not invent citation metadata."),
                HumanMessage(content=f"Question: {query}\n\nDraft answer:\n{output}\n\n"
                             f"Tool used: {fb_name}\nTool result:\n{fb_result}"),
            ])
            output = synthesis.content

    return {"output": output, "intermediate_steps": steps,
            "num_steps": len(steps), "tools_used": tools_used}


def run_cot(query, tools=None, model_name=DEFAULT_LLM_MODEL, temperature=0.0, verbose=False):
    """Chain-of-Thought agent: plan (JSON) -> execute -> synthesize."""
    tools = tools if tools is not None else ALL_TOOLS
    tool_map = _tool_map(tools)
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    plan_text = llm.invoke([
        SystemMessage(content=COT_SYSTEM),
        HumanMessage(content=f"Research question: {query}"),
    ]).content.strip()

    try:
        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0].strip()
        plan = json.loads(plan_text)
    except (json.JSONDecodeError, IndexError):
        q = query.lower()
        if re.search(r"\b(quote|quotes|verbatim|evidence|support|cite)\b", q):
            plan = {"plan": [{"tool": "quote_search", "input": {"query": query}}]}
        elif re.search(r"\b(pdf|document|uploaded|file|summarize|summary)\b", q):
            plan = {"plan": [{"tool": "summarize_active_document", "input": {"query": query}}]}
        else:
            plan = {"plan": [
                {"tool": "web_search", "input": {"query": query}},
                {"tool": "wikipedia_search", "input": {"query": query}},
            ]}

    tool_results, tools_used = [], []
    for step in plan.get("plan", []):
        name = step.get("tool", "")
        tinput = step.get("input", {})
        if name not in tool_map:
            continue
        try:
            fn = tool_map[name]
            if isinstance(tinput, dict) and len(tinput) == 1:
                res = fn.invoke(list(tinput.values())[0])
            elif isinstance(tinput, dict):
                res = fn.invoke(tinput)
            else:
                res = fn.invoke(str(tinput))
            tool_results.append({"tool": name, "input": tinput, "result": res})
            tools_used.append(name)
        except Exception:
            continue

    results_text = "\n\n".join(f"=== {r['tool']}({r['input']}) ===\n{r['result']}" for r in tool_results)
    synthesis = llm.invoke([
        SystemMessage(content="You are a Research Assistant. Given the question and tool results "
                      "below, synthesize a comprehensive answer with citations. Do not make up "
                      "information or citation metadata not in the tool results. Use source titles "
                      "and URLs when available; use n.d. or not identified for missing metadata."),
        HumanMessage(content=f"Question: {query}\n\nTool results:\n{results_text}"),
    ])
    return {
        "output": synthesis.content,
        "num_steps": len(tool_results),
        "tools_used": tools_used,
        "tool_results": tool_results,
        "intermediate_steps": [(r["tool"], r["result"]) for r in tool_results],
    }


def answer(question, strategy="ReAct", tools=None, source_status="", ground=False, **kwargs):
    """Corrected router (#8): the chosen strategy ALWAYS drives execution; a
    specialized tool is suggested as a hint, never a hard bypass.

    If ground=True, run the answer-grounding pass (#1) and attach the report
    under result['grounding'].
    """
    question = normalize_query(question)
    decision = decide_route(question, strategy=strategy, source_status=source_status)
    if decision.hard_error:
        return {"output": decision.hard_error, "tools_used": [], "num_steps": 0,
                "executor": decision.executor, "tool_hint": None}

    agent_input = build_agent_input(question, decision.tool_hint)
    runner = run_react if decision.executor == "react" else run_cot
    result = runner(agent_input, tools=tools, **kwargs)
    result["executor"] = decision.executor
    result["tool_hint"] = decision.tool_hint

    if ground:
        from .grounding import ground_answer

        ground_answer(result, model_name=kwargs.get("model_name", DEFAULT_LLM_MODEL))
    return result

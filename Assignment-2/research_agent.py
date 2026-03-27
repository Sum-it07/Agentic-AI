"""
==============================================================
  Autonomous Research Agent — LangChain + Google Gemini
  Assignment 2 | AI Agent with Web Search & Wikipedia Tools
==============================================================
"""

import os
import re
import datetime
from typing import Optional

# ── LangChain core ────────────────────────────────────────────
# NOTE: In modern LangChain (0.2+), create_react_agent and
#       AgentExecutor live in langchain.agents — but ONLY if
#       langchain-core is also installed correctly.
#       We use the explicit submodule path as a safe fallback.
try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    from langchain.agents.react.agent import create_react_agent   # type: ignore
    from langchain.agents.agent import AgentExecutor              # type: ignore

from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

# ── Gemini LLM ────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI


# ══════════════════════════════════════════════════════════════
#  1.  LLM  –  Google Gemini
# ══════════════════════════════════════════════════════════════
def create_llm(
    model: str = "gemini-3-flash-preview",
    temperature: float = 0.3,
) -> ChatGoogleGenerativeAI:
    """
    Initialise the Google Gemini LLM via LangChain.

    Set GOOGLE_API_KEY in your environment before running:
        Windows CMD  : set GOOGLE_API_KEY=AIza...
        Windows PS   : $env:GOOGLE_API_KEY="AIza..."
        macOS/Linux  : export GOOGLE_API_KEY="AIza..."

    Get a free key at: https://aistudio.google.com/app/apikey
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "\n[ERROR] GOOGLE_API_KEY not found.\n"
            "  Windows CMD : set GOOGLE_API_KEY=AIza...\n"
            "  Windows PS  : $env:GOOGLE_API_KEY='AIza...'\n"
            "  macOS/Linux : export GOOGLE_API_KEY='AIza...'\n"
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,  # Gemini requirement
    )


# ══════════════════════════════════════════════════════════════
#  2.  TOOLS
# ══════════════════════════════════════════════════════════════

def build_tools() -> list:
    """
    Returns a list of tools available to the ReAct agent:
      Tool 1 — DuckDuckGo  : live web search (no API key needed)
      Tool 2 — Wikipedia   : encyclopaedic knowledge base
    """

    # ── Tool 1: Web Search via DuckDuckGo ─────────────────────
    duckduckgo_search = DuckDuckGoSearchRun()

    web_search_tool = Tool(
        name="WebSearch",
        func=duckduckgo_search.run,
        description=(
            "Use this tool to search the web for recent news, statistics, "
            "research papers, and current information about any topic. "
            "Input should be a concise search query string."
        ),
    )

    # ── Tool 2: Wikipedia Knowledge Base ──────────────────────
    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000,
    )
    wiki_run = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wiki_run.run,
        description=(
            "Use this tool to look up background knowledge, definitions, "
            "historical context, and encyclopaedic information on any topic. "
            "Input should be the exact concept or term to look up."
        ),
    )

    return [web_search_tool, wikipedia_tool]


# ══════════════════════════════════════════════════════════════
#  3.  ReAct PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════

REACT_PROMPT_TEMPLATE = """You are an expert research assistant. Your goal is to
thoroughly research the given topic and gather comprehensive information to write
a detailed, structured research report.

You have access to the following tools:

{tools}

Use the following format STRICTLY — do not deviate:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation block can repeat multiple times)
Thought: I now have enough information to write the final comprehensive report
Final Answer: [write the complete structured report here]

Research Guidelines:
- Use WebSearch at least 3 times with different queries
- Use Wikipedia at least 2 times for background context
- Search for: definitions, current stats, real-world examples,
  challenges, expert opinions, and future predictions
- Be thorough — gather data before writing the Final Answer

Begin!

Question: Research the following topic comprehensively and prepare all data
needed for a detailed report: {input}

{agent_scratchpad}"""


def build_react_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=REACT_PROMPT_TEMPLATE,
    )


# ══════════════════════════════════════════════════════════════
#  4.  REPORT FORMATTER
# ══════════════════════════════════════════════════════════════

REPORT_FORMAT_PROMPT = """
Based on all the research data provided below, write a COMPLETE, PROFESSIONAL
research report on the topic: "{topic}"

The report MUST include ALL of these sections, in this exact order:

════════════════════════════════════════════════════════════════
RESEARCH REPORT
════════════════════════════════════════════════════════════════

COVER PAGE
──────────
Title     : [Write a comprehensive, specific title]
Prepared by: Autonomous Research Agent (LangChain + Gemini)
Date      : {date}
Version   : 1.0

════════════════════════════════════════════════════════════════

1. INTRODUCTION
───────────────
Write 2–3 paragraphs covering:
  • What this topic is and why it matters today
  • The current state of the field
  • Scope and purpose of this report

════════════════════════════════════════════════════════════════

2. KEY FINDINGS
───────────────
List 6–7 bullet points. Each point must:
  • Start with a bold label (e.g., **Market Growth:**)
  • Include a specific statistic, fact, or data point
  • Be 2–3 sentences of explanation

════════════════════════════════════════════════════════════════

3. CHALLENGES
─────────────
Describe 4–5 major challenges. For each:
  • Give it a numbered heading
  • Explain what the challenge is (1 sentence)
  • Explain why it is difficult (1–2 sentences)
  • Mention any current efforts to address it (1 sentence)

════════════════════════════════════════════════════════════════

4. FUTURE SCOPE
───────────────
Describe 4–5 forward-looking developments. For each:
  • Give a numbered heading with approximate timeframe
  • Explain what is expected to happen
  • Connect it to current trends from the research

════════════════════════════════════════════════════════════════

5. CONCLUSION
─────────────
Write 2 paragraphs:
  • Paragraph 1: Synthesise the main insights from the report
  • Paragraph 2: Closing perspective on significance and outlook

════════════════════════════════════════════════════════════════

REFERENCES
──────────
List all tools and sources used in bullet format.

════════════════════════════════════════════════════════════════

RESEARCH DATA TO USE:
{research_data}
"""


def format_final_report(
    topic: str,
    raw_research: str,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Send raw agent research to Gemini for clean structured formatting."""
    prompt = REPORT_FORMAT_PROMPT.format(
        topic=topic,
        date=datetime.datetime.now().strftime("%B %d, %Y"),
        research_data=raw_research,
    )
    response = llm.invoke(prompt)
    return response.content


# ══════════════════════════════════════════════════════════════
#  5.  AGENT BUILDER
# ══════════════════════════════════════════════════════════════

def build_agent(llm: ChatGoogleGenerativeAI, tools: list) -> AgentExecutor:
    """Create and return a ReAct AgentExecutor."""
    prompt = build_react_prompt()

    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,               # shows Thought/Action/Observation loop
        max_iterations=14,
        max_execution_time=240,     # 4-minute timeout
        handle_parsing_errors=True, # auto-recover from malformed outputs
        return_intermediate_steps=True,
    )
    return executor


# ══════════════════════════════════════════════════════════════
#  6.  SAVE REPORT
# ══════════════════════════════════════════════════════════════

def save_report(report: str, topic: str, output_dir: str = ".") -> str:
    """Save the formatted report to a Markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"{output_dir}/report_{safe_name}_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    return filename


# ══════════════════════════════════════════════════════════════
#  7.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run_research_agent(
    topic: str,
    save_output: bool = True,
    output_dir: str = "reports",
    model: str = "gemini-3-flash-preview",
) -> dict:
    """
    Full 3-step pipeline:
      1. Initialise Gemini LLM + Tools
      2. Run ReAct Agent to gather research
      3. Format raw research into a structured report
      4. (Optional) Save to .md file
    """

    print("\n" + "═" * 60)
    print("  🔍  Autonomous Research Agent  |  LangChain + Gemini")
    print("═" * 60)
    print(f"  Topic  : {topic}")
    print(f"  Model  : {model}")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60 + "\n")

    # ── Step 1: Initialise ───────────────────────────────────
    print("▶  Step 1/3 — Initialising LLM and tools …")
    llm   = create_llm(model=model)
    tools = build_tools()
    agent = build_agent(llm, tools)
    print(f"   ✓ LLM   : {model}")
    print(f"   ✓ Tools : {[t.name for t in tools]}\n")

    # ── Step 2: Run ReAct Agent ──────────────────────────────
    print("▶  Step 2/3 — Running ReAct research loop …\n")
    result = agent.invoke({"input": topic})

    raw_output = result.get("output", "")

    # Collect all tool observations for richer report context
    intermediate = result.get("intermediate_steps", [])
    tool_logs = "\n\n".join(
        f"[Tool: {action.tool}]\n"
        f"Query: {action.tool_input}\n"
        f"Result:\n{obs}"
        for action, obs in intermediate
    )
    full_context = f"{tool_logs}\n\n--- Agent Final Summary ---\n{raw_output}"

    # ── Step 3: Format Report ────────────────────────────────
    print("\n▶  Step 3/3 — Formatting structured report …")
    final_report = format_final_report(topic, full_context, llm)
    print("   ✓ Report formatted\n")

    # ── Save ─────────────────────────────────────────────────
    saved_path: Optional[str] = None
    if save_output:
        saved_path = save_report(final_report, topic, output_dir)
        print(f"   💾  Saved → {saved_path}\n")

    print("═" * 60)
    print("  ✅  Research Complete!")
    print("═" * 60 + "\n")

    return {
        "report":       final_report,
        "raw_research": raw_output,
        "saved_path":   saved_path,
    }


# ══════════════════════════════════════════════════════════════
#  8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — LangChain + Gemini"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Impact of AI in Healthcare",
        help="Topic to research (wrap in quotes if multi-word)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        choices=["gemini-3-flash-preview", "gemini-3.5-sonnet", "gemini-2.0-flash"],
        help="Gemini model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Folder to save reports (created if absent)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print report to console only, skip file save",
    )

    args = parser.parse_args()

    output = run_research_agent(
        topic=args.topic,
        save_output=not args.no_save,
        output_dir=args.output_dir,
        model=args.model,
    )

    print("\n" + "═" * 60)
    print("  GENERATED REPORT")
    print("═" * 60 + "\n")
    print(output["report"])

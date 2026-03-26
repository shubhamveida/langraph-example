import os
import warnings
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
search_tool = TavilySearch(max_results=3)

class AgentState(TypedDict):
    ticker: str
    research_notes: List[str]
    sentiment: str
    loop_count: int

def research_node(state: AgentState):
    ticker = state["ticker"]
    print(f"🌐 [Tavily] Fetching live 2026 data for {ticker}...")
    try:
        search_query = f"latest financial news and stock analysis for {ticker} March 2026"
        results = search_tool.invoke({"query": search_query})
        if isinstance(results, str): results = [{"content": results}]
        new_notes = [str(r.get('content', r)) for r in results]
        return {"research_notes": state.get("research_notes", []) + new_notes, "loop_count": state.get("loop_count", 0) + 1}
    except Exception as e:
        return {"research_notes": [f"Error: {e}"], "loop_count": state["loop_count"] + 1}

def analysis_node(state: AgentState):
    print("🧠 [Gemini 3] Compressing analysis...")
    combined_notes = "\n".join(state["research_notes"])
    
    # NEW CONCISE PROMPT
    prompt = (
        f"Analyze {state['ticker']} based on March 2026 data.\n"
        "FORMATTING RULES:\n"
        "1. Start with 'VERDICT: [BUY/SELL/HOLD]'\n"
        "2. Follow with 'REASONING:' and exactly 4-5 bullet points or lines of text.\n"
        "3. Keep it extremely punchy. No fluff.\n"
        f"Data: {combined_notes}"
    )
    
    response = llm.invoke(prompt)
    raw_content = response.content
    if isinstance(raw_content, list):
        text_content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content])
    else:
        text_content = str(raw_content)
    
    return {"sentiment": text_content.strip()}

def should_continue(state: AgentState):
    if "UNKNOWN" in state["sentiment"].upper() and state["loop_count"] < 2:
        return "retry"
    return "end"

# Graph Setup
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("analyzer", analysis_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyzer")
workflow.add_conditional_edges("analyzer", should_continue, {"retry": "researcher", "end": END})
app = workflow.compile()

def run_audit_system():
    while True:
        print("\n" + "═"*50 + "\n🚀 QUICK-STRIKE ANALYST (v2.1)\n" + "═"*50)
        ticker = input("Enter Ticker (or 'quit'): ").strip().upper()
        if ticker in ['QUIT', 'EXIT', 'Q']: break
        if not ticker: continue

        try:
            final_output = app.invoke({"ticker": ticker, "research_notes": [], "loop_count": 0})
            print(f"\n█ REPORT: {ticker}\n{final_output['sentiment']}\n" + "─"*50)
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_audit_system()


    
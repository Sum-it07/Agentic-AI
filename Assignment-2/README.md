# 🤖 Autonomous Research Agent
### LangChain + Google Gemini | Assignment 2

---

## Architecture Overview

```
User Input (Topic)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                          │
│                                                              │
│   Thought → Action → Observation → Thought → ...            │
│                                                              │
│   ┌─────────────────┐    ┌──────────────────────┐           │
│   │  Tool 1         │    │  Tool 2              │           │
│   │  WebSearch      │    │  Wikipedia           │           │
│   │  (DuckDuckGo)   │    │  (Knowledge Base)    │           │
│   └─────────────────┘    └──────────────────────┘           │
│                                                              │
│              LLM: Google Gemini 3 Preview                    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  Raw Research Data
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│            Report Formatter (Gemini)                         │
│   • Cover Page    • Introduction    • Key Findings           │
│   • Challenges    • Future Scope    • Conclusion             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  📄 reports/report_<topic>_<timestamp>.md
```

---

## Setup

### 1. Clone / copy files
```bash
mkdir research_agent && cd research_agent
# copy research_agent.py, requirements.txt here
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Google Gemini API key
```bash
export GOOGLE_API_KEY="sk-gemini-api03-..."
```
Or create a `.env` file:
```
GOOGLE_API_KEY=sk-gemini-api03-...
```

---

## Usage

### Default run (AI in Healthcare)
```bash
python research_agent.py
```

### Custom topic
```bash
python research_agent.py --topic "Quantum Computing and Cryptography"
```

### Additional options
```bash
python research_agent.py \
  --topic "Climate Change and Renewable Energy" \
       --model "gemini-3-flash-preview" \
  --output-dir "my_reports" \
  --no-save          # print to console only
```

---

## Output

Reports are saved to `reports/report_<topic>_<timestamp>.md` and contain:

| Section | Content |
|---------|---------|
| 📄 Cover Page | Title, author, date, version |
| 📝 Introduction | Context, scope, purpose |
| 🔍 Key Findings | 5–7 substantive insights with data |
| ⚠️ Challenges | 4–5 barriers and limitations |
| 🚀 Future Scope | 4–5 forward-looking predictions |
| ✅ Conclusion | Synthesis and closing perspective |
| 📚 References | All sources consulted |

---

## Key Components

| Component | Implementation |
|-----------|----------------|
| **Framework** | LangChain 0.3+ |
| **LLM** | Google Gemini 3 Flash (`gemini-3-flash-preview`) |
| **Agent Type** | ReAct (Reason + Act) |
| **Tool 1** | `DuckDuckGoSearchRun` — live web search |
| **Tool 2** | `WikipediaQueryRun` — encyclopaedic knowledge |
| **Max Iterations** | 12 reasoning steps |
| **Timeout** | 180 seconds |

---

## How the ReAct Loop Works

```
Thought: I need to research AI applications in radiology.
Action: WebSearch
Action Input: "AI radiology diagnosis accuracy 2024 statistics"
Observation: [search results...]

Thought: I should get background context from Wikipedia.
Action: Wikipedia
Action Input: "Artificial intelligence in healthcare"
Observation: [wikipedia article...]

Thought: I need recent market data.
Action: WebSearch
Action Input: "AI healthcare market size 2024 forecast"
Observation: [search results...]

... (continues for 8-12 steps)

Thought: I now have enough information to write the report.
Final Answer: [complete structured report]
```

---

## File Structure

```
research_agent/
│
├── research_agent.py          # Main agent code
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── sample_report_ai_healthcare.md  # Example output
└── reports/                   # Generated reports (auto-created)
    └── report_<topic>_<timestamp>.md
```

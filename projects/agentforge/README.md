# AgentForge — Multi-Agent Research Assistant

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Powered-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker" />
</p>

## 🎯 Problem Statement

Single-prompt LLMs can't handle complex research tasks that require planning, searching multiple sources, synthesizing findings, and fact-checking. AgentForge uses **LangGraph** to orchestrate specialized AI agents that collaborate through a stateful graph — just like a real research team.

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │   User      │
                    │   Query     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Router    │  Classifies query type
                    │   Agent     │  and delegates to the
                    └──┬───┬───┬──┘  right workflow
                       │   │   │
          ┌────────────┘   │   └────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
   │ Researcher  │  │ Researcher │  │  Direct     │
   │ (Web)       │  │ (Document) │  │  Answer     │
   └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
          │                │                │
          └────────┬───────┘                │
                   │                        │
            ┌──────▼──────┐                 │
            │   Writer    │  Synthesizes    │
            │   Agent     │  research into  │
            └──────┬──────┘  a report       │
                   │                        │
            ┌──────▼──────┐                 │
            │   Critic    │  Fact-checks    │
            │   Agent     │  and scores     │
            └──────┬──────┘  quality        │
                   │                        │
              ┌────▼────┐                   │
              │ Quality │     YES           │
              │ ≥ 0.8?  ├──────────────┐    │
              └────┬────┘              │    │
                   │ NO (retry)        │    │
                   │                   │    │
            ┌──────▼──────┐    ┌───────▼────▼──┐
            │   Writer    │    │    Human-in-  │
            │   (Revise)  │    │    the-Loop   │
            └─────────────┘    │    Approval   │
                               └──────┬───────┘
                                      │
                               ┌──────▼──────┐
                               │   Final     │
                               │   Output    │
                               └─────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **LangGraph State Machine** | Stateful, cyclical graph with conditional routing — not simple chain |
| **4 Specialized Agents** | Router, Researcher, Writer, Critic — each with focused expertise |
| **Tool Use** | Web search via Tavily, document analysis, calculations |
| **Conditional Routing** | Dynamic agent selection based on query classification |
| **Retry Loop** | Automatic revision cycle when quality score < threshold |
| **Human-in-the-Loop** | Optional approval step before delivering final output |
| **Streaming** | Real-time agent activity visualization |
| **Error Recovery** | Graceful fallbacks when tools fail |

## 🚀 Quick Start

```bash
git clone https://github.com/sulaxmi22/agentforge.git
cd agentforge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python -m uvicorn backend.main:app --reload --port 8001
```

## 📁 Project Structure

```
agentforge/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── config.py             # Configuration
│   ├── agents/
│   │   ├── graph.py          # LangGraph state machine
│   │   ├── state.py          # Typed state definitions
│   │   ├── nodes.py          # Agent node implementations
│   │   └── tools.py          # Tool definitions
├── frontend/
│   ├── index.html            # Agent visualization UI
│   └── styles.css
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 📝 License

MIT License

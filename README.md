# 🔬 Autonomous Research Agent

A production-grade autonomous multi-agent system for analyzing research papers from arXiv using **LangGraph**, **FastAPI**, and **Ollama (Llama 3)**.

## ✨ Features

- **🆓 100% Local & Free** - Runs entirely on Ollama (no API keys required)
- **🤖 Multi-Agent Architecture** - Researcher → Analyzer → Comparator → Synthesizer
- **🔬 Scientific Integrity** - Negative constraints prevent fabricated contradictions
- **💾 Persistent Storage** - SQLite database with job history and caching
- **📊 Metric Comparison** - Automatically extracts and compares quantitative metrics
- **📚 Job History** - View, reload, and manage past research jobs
- **🌐 REST API** - FastAPI backend with async job processing
- **🎨 Web UI** - Streamlit interface for easy interaction
- **📖 Citation Tracking** - Every finding includes proper citations

## 🏗️ Architecture

```
┌──────────────┐         ┌──────────────┐
│  Streamlit   │  HTTP   │   FastAPI    │
│  Frontend    │────────▶│   Backend    │
└──────────────┘         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │   LangGraph Workflow  │
                    └───────────┬───────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   Researcher   Analyzer   Comparator  Synthesizer    END
        │           │           │           │
        ▼           ▼           ▼           ▼
   arXiv PDFs  Patterns &  Metric Table  MD Report
               FAISS Index  Comparison    + JSON
```

### Components

- **MCP arXiv Tool** - Fetches and downloads research papers
- **FAISS Vector Store** - Local semantic search with metadata
- **LangGraph** - Orchestrates agent workflow
- **Ollama (Llama 3)** - Local LLM for analysis
- **Sentence Transformers** - Local embeddings (no API)

## 📦 Installation

### Prerequisites

1. **Python 3.13+**
2. **Ollama** - [Install from ollama.com](https://ollama.com)

### Setup

```bash
# 1. Clone or navigate to project
cd autonomous-research-agent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Llama 3 model
ollama pull llama3.1

# 5. Start Ollama server (in separate terminal)
ollama serve
```

## 🚀 Usage

### Quick Start

**Terminal 1 - Start Backend:**
```bash
uvicorn api.api:app --reload --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run app.py
```

Then open browser to `http://localhost:8501`

### API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Command Line Interface

```bash
# Basic usage
python main.py "quantum computing error correction"

# Specify number of papers
python main.py "solid-state batteries" --papers 10
```

### API Endpoints

- `POST /research` - Create new research job
- `GET /status/{job_id}` - Get job status and progress
- `GET /results/{job_id}` - Retrieve completed results
- `GET /jobs` - List all jobs (history)
- `DELETE /jobs/{job_id}` - Delete a job
- `GET /health` - API health check

## 📊 Output

### Markdown Report (`outputs/{job_id}/report.md`)

- Executive Summary
- **Quantitative Metrics Comparison** (NEW!)
  - Automatically extracted metrics table
  - Energy density, cycle life, conductivity, etc.
  - Cross-paper comparison
- Key Findings with citations
- Contradictions & Debates (only if TRUE contradictions exist)
- Complementary Findings
- Emerging Trends
- Research Gaps
- Full References

### JSON Insights (`outputs/{job_id}/insights.json`)

```json
{
  "topic": "solid-state batteries",
  "timestamp": "2026-01-03T14:30:00",
  "papers_analyzed": 7,
  "key_findings": [...],
  "contradictions": [...],
  "complementary_findings": [...],
  "trends": [...],
  "comparison": {
    "metrics_table": [...],
    "metric_names": ["Energy Density", "Ionic Conductivity"],
    "comparison_summary": "..."
  }
}
```

## 🔬 Scientific Integrity

### Negative Constraint for Contradictions

The Analyzer Agent uses a **negative constraint prompt** to prevent fabricating conflicts:

- **NEVER** invents contradictions
- Forces categorical responses: CONTRADICTION | COMPLEMENTARY | UNRELATED
- Defaults to "COMPLEMENTARY" when papers address different aspects
- Reports "No contradictions found" if none exist

This ensures the system doesn't "make up fights" between scientists.

## 🏛️ Project Structure

```
autonomous-research-agent/
├── api/                        # FastAPI Backend
│   ├── api.py                  # REST API endpoints
│   ├── schemas.py              # Pydantic models
│   ├── job_store.py            # SQLite job persistence
│   └── research_worker.py      # Background task executor
├── agents/                     # LangGraph Agents
│   ├── researcher.py           # Fetches papers, builds FAISS
│   ├── analyzer.py             # Finds patterns, contradictions
│   ├── comparator.py           # Extracts quantitative metrics
│   └── synthesizer.py          # Generates reports
├── graph/                      # LangGraph Workflow
│   ├── state.py                # TypedDict schemas
│   └── research_graph.py       # Workflow orchestration
├── mcp/                        # Model Context Protocol Tools
│   └── arxiv_tool.py           # arXiv paper fetching
├── memory/                     # Storage & Retrieval
│   ├── vector_store.py         # FAISS vector database
│   ├── pdf_extractor.py        # PDF text extraction
│   ├── pdfs/{job_id}/          # Cached PDFs per job
│   └── faiss/{job_id}/         # FAISS indices per job
├── outputs/{job_id}/           # Job-specific outputs
│   ├── report.md               # Generated markdown report
│   └── insights.json           # Structured JSON insights
├── app.py                      # Streamlit frontend
├── main.py                     # CLI interface
├── config.py                   # Configuration settings
├── jobs.db                     # SQLite database
└── requirements.txt            # Python dependencies
```

## ⚙️ Configuration

Edit `config.py` or use environment variables:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"
LLM_TEMPERATURE = 0.7
MAX_PAPERS = 7
CHUNK_SIZE = 700  # Optimized for arXiv PDFs
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## 🔧 Key Engineering Decisions

### 1. PDF Chunking Strategy

Uses **RecursiveCharacterTextSplitter** (not simple sentence splitting):

- **Chunk Size:** 700 tokens (sweet spot for Llama 3)
- **Overlap:** 100 tokens (~14%)
- **Separators:** `["\n\n", "\n", ". ", " ", ""]`

This handles multi-column layouts, formulas, and bibliographies in arXiv PDFs.

### 2. Metadata Preservation

Every FAISS chunk carries:
- `arxiv_id` - Paper identifier
- `citation` - Pre-formatted citation string
- `chunk_id` - Position within paper

This prevents "metadata loss" - the #1 silent failure in RAG systems.

### 3. MCP Architecture

Tools follow Model Context Protocol:
- Self-contained with clear inputs/outputs
- Stateless operation
- Observable/loggable

## 📈 Performance

- **Execution Time:** <5 minutes for 7 papers
- **Cost:** $0 (fully local)
- **Privacy:** 100% - no data leaves your machine

## 🛠️ Development

### Run Tests

```bash
pytest tests/
```

### View Logs

```bash
tail -f logs/research_*.log
```

## 🎯 Use Cases

- **Literature Review** - Quickly synthesize findings across papers
- **Contradiction Detection** - Find conflicting research claims
- **Trend Analysis** - Identify emerging research directions
- **Gap Analysis** - Discover understudied areas
- **Citation Management** - Track sources automatically

## 🤝 Contributing

This is a reference implementation demonstrating:
- LangGraph multi-agent orchestration
- MCP tool integration
- Local-first AI architecture
- Scientific integrity in LLM outputs

## 📝 License

MIT

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Backend API framework
- [Streamlit](https://streamlit.io/) - Web UI framework
- [Ollama](https://ollama.com) - Local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [arXiv API](https://arxiv.org/help/api) - Research paper access

---

**Note:** This system is designed for research assistance. Always verify findings against original sources.

# Product Review MAS (CrewAI + Ollama)

Local multi-agent pipeline: load reviews → statistics → fraud heuristics → recommendation. **No cloud LLM APIs**—only [Ollama](https://ollama.com) on your machine.

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running ([download](https://ollama.com/download))

## Setup (first time)

### 1. Create a virtual environment

**Windows (PowerShell)**

```powershell
cd "path\to\this\project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
cd path/to/this/project
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the chat model

Default model is **`llama3.1:8b`** (works well with tool calling). From a terminal:

```bash
ollama pull llama3.1:8b
```

To use another local model:

- **Windows (PowerShell):** `$env:MAS_OLLAMA_MODEL = "your-model-name"`
- **Windows (cmd):** `set MAS_OLLAMA_MODEL=your-model-name`
- **macOS / Linux:** `export MAS_OLLAMA_MODEL=your-model-name`

## Run the pipeline

From the project root with the venv activated:

```bash
python main.py
```

Optional:

```bash
python main.py --review-file sample_reviews.json
```

Outputs:

- Console logs
- **`last_pipeline_state.json`** — full run snapshot (`tool_calls`, `execution_trace`, stages, etc.)
- **`logs/system.log`** — file log

## Demo UI (Streamlit)

Optional browser UI that calls the same `run_pipeline` entry point as the CLI (no changes to orchestration logic). From the project root with the **venv activated**, install deps once (**`pip install -r requirements.txt`** — this includes Streamlit). Then use **Python’s module mode** so Streamlit runs with the same interpreter as CrewAI (avoids `ModuleNotFoundError: crewai` when a global `streamlit` is on `PATH`, and avoids `No module named streamlit` if Streamlit was never installed in the venv):

```bash
python -m streamlit run demo_app.py
```

Ollama should be running for full pipeline runs (unless you use the sidebar “Tools-only run” option). Telemetry prompts are disabled for this repo via `.streamlit/config.toml`.

The app writes **`last_pipeline_state.json`** after a successful run, matching the CLI snapshot format.

## Verify everything works

1. **Assignment checks** (expects Ollama up and the model pulled):

   ```bash
   python evaluate_pipeline.py
   ```

   You want **Overall: PASS** in the logs (`logs/evaluation.log`).

2. **Tests** (no live LLM):

   ```bash
   python -m pytest -m "not integration"
   ```

3. **Full integration** (optional, needs Ollama):

   ```bash
   python -m pytest tests/test_system.py::test_system_end_to_end_pipeline -m integration
   ```

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI entry |
| `demo_app.py` | Streamlit demo (optional) |
| `coordinator.py` | Orchestration, shared state, Crew phases |
| `evaluate_pipeline.py` | Grading-style checks |
| `agents/` | Four CrewAI agents (one tool each) |
| `tools/` | Custom tools + `crew_tool_factories.py` |
| `validation/` | Pydantic schemas and stage validators |
| `tests/` | Pytest suite |
| `sample_reviews.json` | Example input |
| `CONTRIBUTORS.md` | Team ↔ agent ↔ tool ↔ test mapping |


## Team contributions

See **`CONTRIBUTORS.md`** and replace placeholder student names with your roster.

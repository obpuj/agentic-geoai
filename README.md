# Agentic GeoAI Framework

An agent-oriented GeoAI prototype for querying geospatial analysis results through a natural-language interface.

> **Current implementation note:** The current runtime uses precomputed GeoTIFF outputs for Vision analysis. The GIS agent is currently a stub and is being upgraded toward real OSM-based analysis. SegFormer/Prithvi are used in the offline processing pipeline rather than automatically running for every web request.

## Current Runtime

```text
User Query
    ↓
Flask Web App
    ↓
LLM Planner
    ↓
Gazetteer + Temporal Resolution
    ↓
Router
    ├── Vision Agent → Precomputed GeoTIFF analysis
    └── GIS Agent    → Currently stubbed
    ↓
Reporter
    ├── Knowledge Base / RAG
    └── Reporter LLM
    ↓
JSON Response
```

## Installation

### 1. Prerequisites

Install:
- Python
- Git
- Git LFS

Verify:

```powershell
python --version
git --version
git lfs version
```

Initialize Git LFS if necessary:

```powershell
git lfs install
```

### 2. Clone

```powershell
git clone <REPOSITORY_URL>
cd agentic-geoai
git lfs pull
```

Check:

```powershell
git lfs ls-files
```

### 3. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation is blocked, install using the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Rasterio fails on Windows, use a compatible conda/conda-forge environment or capture the exact installation error before changing GDAL versions.

### 5. Configure the LLM

Create `.env` in the repository root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

or:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit `.env`.

Use `.env.example` for the safe template:

```text
GROQ_API_KEY=
GEMINI_API_KEY=
```

Check that a key is loaded without printing it:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('GROQ_API_KEY') or os.getenv('GEMINI_API_KEY')))"
```

### 6. Run

From the project root:

```powershell
python -m auie.webapp
```

Open:

```text
http://127.0.0.1:5050
```

Stop with `CTRL+C`.

## First Run

The first query can be slower because the Sentence Transformer embedding model may need to be downloaded and initialized. This is separate from the Groq/Gemini API.

## Data

### Precomputed raster data

```text
data/precomputed/
```

These files may be managed by Git LFS:

```powershell
git lfs pull
```

### Knowledge Base

```text
data/kb/
├── vecs.npy
└── chunks.jsonl
```

### Gazetteer

```text
data/BBMP_oldWards.geojson
```

## ML Architecture

SegFormer and Prithvi are not currently executed for every browser query.

```text
SegFormer / Prithvi
        ↓
Offline inference
        ↓
Precomputed GeoTIFF
        ↓
Vision Agent
        ↓
Query-time clipping/statistics
```

This keeps the web application lighter and faster than running large remote-sensing models on every request.

## Troubleshooting

### Missing `sentence_transformers`

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Run:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify:

```powershell
python -c "import sentence_transformers; print('OK')"
```

### Rasterio installation failure

Rasterio has compiled geospatial dependencies. If the normal installation fails, use a compatible conda/conda-forge setup rather than randomly installing GDAL builds.

### PyTorch installation issues

`sentence-transformers` uses the PyTorch/Transformers stack. If a CPU-only installation is appropriate, use the official PyTorch installation selector for the machine rather than copying a CUDA-specific command.

### Groq 401

Check whether the environment variable is loaded:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('GROQ_API_KEY')))"
```

Do not print the API key itself.

### Groq 404

Check the model configured in:

```text
auie/llm.py
```

The configured model must still be available through the Groq endpoint.

### Port 5050 already in use

```powershell
netstat -ano | findstr :5050
```

Then, if appropriate:

```powershell
taskkill /PID <PID> /F
```

Restart:

```powershell
python -m auie.webapp
```

### Missing Git LFS files

```powershell
git lfs install
git lfs pull
git lfs ls-files
```

## Project Structure

```text
agentic-geoai/
│
├── auie/
│   ├── webapp.py
│   ├── pipeline.py
│   ├── planner.py
│   ├── router.py
│   ├── reporter.py
│   ├── knowledge.py
│   ├── gazetteer.py
│   ├── temporal.py
│   ├── llm.py
│   └── agents/
│       ├── vision.py
│       └── base.py
│
├── data/
│   ├── kb/
│   ├── precomputed/
│   └── BBMP_oldWards.geojson
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Development Roadmap

1. Freeze current working baseline.
2. Replace `StubGISAgent` with real OSM-based analysis.
3. Improve Vision provenance using `meta.json`.
4. Make Knowledge/RAG respond to actual documentation/provenance questions.
5. Extend Planner output only as needed for routing.
6. Make Router support meaningful combinations of Vision, GIS and Knowledge.
7. Make `EvidenceBundle` traceable and auditable.
8. Add unit/integration tests.
9. Update final documentation to describe only implemented behavior.

## Documentation Principle

The project distinguishes between:

- **Implemented:** executes at runtime.
- **Offline:** used to generate artifacts before the web application runs.
- **Planned:** intended functionality not yet implemented.

This distinction should be preserved in future documentation.

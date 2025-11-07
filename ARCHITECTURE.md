## Streamlit Fleet Transition Toolkit

### Overview
The application is a Streamlit-hosted decision support tool that lets fleet planners build scenarios, explore the supplied CSV datasets, and quantify the operational, economic, and environmental outcomes of different energy pathways. It also includes an AI assistant for natural-language guidance and a PDF exporter for curated reports.

### Key Components
- **streamlit_app.py** — Streamlit entry point that orchestrates UI, scenario management, data preview, calculators, AI assistant, and export workflows.
- **app/models.py** — Dataclass models describing fleet scenarios, energy mixes, and structured result payloads.
- **app/constants.py** — Baseline assumptions for vehicle archetypes, cost factors, emission factors, and sensitivity defaults.
- **app/calculators.py** — Core analytics for energy demand, economic evaluation (capex/opex/TCO), environmental metrics (well-to-wheel, intensity, abatement), and sensitivity bands.
- **app/data.py** — Utilities for loading bundled CSVs, accepting user uploads, and preparing tidy dataframes for analysis or lookup.
- **app/ai.py** — Wrapper around the OpenAI API (or any compatible client) to provide contextualised responses while gracefully degrading if no API key is configured.
- **app/report.py** — PDF report builder that composes narrative sections, tables, and generated figures for download through Streamlit.

### Data Flow
1. Users load or upload datasets via Streamlit (`app/data.py`), which are cached for re-use.
2. Fleet scenarios are captured through sidebar forms (`app/models.py`) and stored in `st.session_state`.
3. Calculators consume each scenario and produce structured outputs (`app/calculators.py`), including sensitivity and baseline comparisons.
4. Visualisations and result tables are rendered in the main pane; `app/report.py` re-composes them into a downloadable PDF.
5. The AI assistant (`app/ai.py`) receives a compact summary of active scenarios and recent insights to contextualise user prompts.

### Testing Strategy
- Unit tests for `app/calculators.py` covering energy demand, cost, and emission outputs, including edge cases.
- Fixture-backed tests for input validation in `app/models.py`.
- Lightweight smoke test for `app/report.py` to ensure PDFs are generated without errors when provided mock data.

### Configuration & Secrets
- Dependencies managed via `requirements.txt` (Streamlit, pandas, numpy, plotly, matplotlib, fpdf2, openai, pytest).
- OpenAI API key supplied at runtime using an environment variable (`OPENAI_API_KEY`), ensuring the app runs in read-only mode when absent.


# AI-SSP

Proof-of-concept exploring how IPCC Shared Socioeconomic Pathways (SSPs) can provide a useful framework for AI safety and governance evaluations.

🏆 Awarded 2nd Place – [Apart Research AI Forecasting Hackathon](https://apartresearch.com/sprints/the-ai-forecasting-hackathon-2025-10-31-to-2025-11-02) (November 2025).

[Read the full write-up (PDF)](ai_ssp.pdf).

The dashboard links SSP global metrics with simple AI risk indicators for side‑by‑side comparison.

Real data has been sourced from [Our World in Data's IPCC Scenarios Data Explorer](https://ourworldindata.org/explorers/ipcc-scenarios), originally extracted from [the SSP database](https://tntcat.iiasa.ac.at/SspDb/dsd).

AI metrics and risk scores should not be considered as forecasts, but as illustrative placeholders. This short weekend project aims to show a possible framework for situating AI risks within wider social and climate dynamics and to prompt more rigorous follow‑up work.

## Usage
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Launch the dashboard from the project root:

```bash
python -m streamlit run app.py
```

# AI-SSP

Proof-of-concept exploring how IPCC Shared Socioeconomic Pathways (SSPs) can provide a useful framework for AI safety and governance evaluations.

The dashboard links SSP global metrics with simple AI risk indicators for side‑by‑side comparison.

Real data has been source from [Our World in Data's IPCC Scenarios Data Explorer](https://ourworldindata.org/explorers/ipcc-scenarios), originally extracted from [the SSP database](https://tntcat.iiasa.ac.at/SspDb/dsd). 

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

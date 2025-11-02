"""
AI-SSP Dashboard: Interactive Visualization of AI Development Scenarios
Based on IPCC Shared Socioeconomic Pathways
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="AI-SSP Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AI-specific projections (expert judgment based on SSP narratives and AI forecasting literature)
# These represent our best estimates for AI development trajectories under each SSP
AI_PROJECTIONS = {
    'years': [2020, 2030, 2040],  # Kept for alignment with baseline socioeconomic data
    # Single scalar: Safety research investment share in 2040 (or representative level across decade window)
    'safety_research_pct': {
        'SSP1': 30,   # High safety prioritization
        'SSP2': 15,   # Moderate safety investment
        'SSP3': 5,    # Minimal safety focus
        'SSP4': 18,   # Elite safety for elite systems
        'SSP5': 12    # Tech-optimist, lower relative safety share
    }
}

# Risk probabilities by 2040
RISK_DATA = {
    # Probabilities or relative concern levels by 2040 (0-100 scale) for each SSP
    'risks': [
        'Economic\nCollapse',         # Rapid automation leading to severe economic dislocation
        'Power\nConflict',            # AI-enabled interstate or bloc power escalation
        'Totalitarian\nLock-in',      # Permanent algorithmic authoritarian control
        'CBRN\nMisuse',               # Chemical/Biological/Radiological/Nuclear enabled misuse
        'Existential\nCatastrophe'    # Irreversible loss of long-term future potential
    ],
    # Placeholder values (tunable): lower across cooperative, higher in fragmentation
    'SSP1': [15, 15, 10, 12, 8],
    'SSP2': [35, 40, 25, 30, 20],
    'SSP3': [70, 85, 60, 65, 55],
    'SSP4': [55, 50, 65, 45, 35],
    'SSP5': [45, 55, 30, 35, 25]
}


# Scenario definitions
SCENARIOS = {
    'SSP1': {
        'name': 'SSP1: Sustainability',
        'color': '#2ecc71'
    },
    'SSP2': {
        'name': 'SSP2: Middle of the Road',
        'color': '#3498db'
    },
    'SSP3': {
        'name': 'SSP3: Regional Rivalry',
        'color': '#e74c3c'
    },
    'SSP4': {
        'name': 'SSP4: Inequality',
        'color': '#f39c12'
    },
    'SSP5': {
        'name': 'SSP5: Fossil-Fueled Development',
        'color': '#9b59b6'
    }
}


@st.cache_resource(show_spinner=False)
def fetch_ipcc_table():
    """Fetch the raw IPCC scenarios table once (network + disk heavy). Cached across reruns."""
    from owid.catalog import find
    return find("ipcc_scenarios", channels=["grapher"]).iloc[0].load()


@st.cache_data(show_spinner=False)
def load_owid_data() -> Optional[pd.DataFrame]:
    """Load & clean baseline SSP data (2020, 2030, 2040) from cached raw table."""
    try:
        tb_raw = fetch_ipcc_table()
        tb = tb_raw.reset_index().rename(columns={"country": "scenario"}).drop_duplicates()
        baseline_scenarios = [
            'SSP1 - Baseline', 'SSP2 - Baseline', 'SSP3 - Baseline',
            'SSP4 - Baseline', 'SSP5 - Baseline'
        ]
        tb_filtered = tb[tb['scenario'].isin(baseline_scenarios) & tb['year'].isin([2020, 2030, 2040])].copy()
        tb_filtered['ssp'] = tb_filtered['scenario'].str.extract(r'(SSP\d)')[0]
        columns_to_keep = {
            'gdp_per_capita__region_global': 'gdp_per_capita',
            'population__region_global': 'population_people',
            'emissions_co2__region_global': 'co2_emissions_tonnes',
            'final_energy__region_global': 'final_energy',
            'elec_capacity__region_global': 'electricity_capacity'
        }
        base_cols = ['ssp','year']
        available_cols = [c for c in columns_to_keep if c in tb_filtered.columns]
        tb_clean = tb_filtered[base_cols + available_cols].copy()
        rename_map = {k: v for k, v in columns_to_keep.items() if k in available_cols}
        tb_clean = tb_clean.rename(columns=rename_map)
        if 'population_people' in tb_clean.columns:
            tb_clean['population_million'] = tb_clean['population_people'] / 1e6
            tb_clean = tb_clean.drop(columns=['population_people'])
        if 'co2_emissions_tonnes' in tb_clean.columns:
            tb_clean['co2_emissions_gt'] = tb_clean['co2_emissions_tonnes'] / 1e9
            tb_clean = tb_clean.drop(columns=['co2_emissions_tonnes'])
        return tb_clean
    except Exception as e:
        st.warning(f"Could not load OWID data: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_combined_data(_owid_data: Optional[pd.DataFrame] = None):
    """Strictly merge cleaned OWID baseline data with AI projections; no synthetic fallbacks."""
    if _owid_data is None:
        raise ValueError("OWID IPCC data not loaded; cannot build baseline metrics.")

    required_cols = ['ssp','year','gdp_per_capita','population_million','co2_emissions_gt','final_energy','electricity_capacity']
    missing_global = [c for c in required_cols if c not in _owid_data.columns]
    if missing_global:
        raise KeyError(f"Missing required columns in OWID dataset: {missing_global}")

    years_sorted = sorted(list(set(_owid_data['year'])))
    combined = {
        'years': years_sorted,
        'gdp_per_capita': {},
        'safety_research_pct': {},
        'population': {},
        'co2_emissions': {},
        'final_energy': {},
        'electricity_capacity': {}
    }

    for ssp in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        ssp_rows = _owid_data[_owid_data['ssp'] == ssp].sort_values('year')
        if ssp_rows.empty:
            raise ValueError(f"No rows for {ssp} in OWID data.")
        # Expand scalar safety value across available years for consistency in charting interfaces if needed
        safety_scalar = AI_PROJECTIONS['safety_research_pct'][ssp]
        combined['safety_research_pct'][ssp] = [safety_scalar] * len(years_sorted)
        combined['gdp_per_capita'][ssp] = ssp_rows['gdp_per_capita'].tolist()
        combined['population'][ssp] = ssp_rows['population_million'].tolist()
        combined['co2_emissions'][ssp] = ssp_rows['co2_emissions_gt'].tolist()
        combined['final_energy'][ssp] = ssp_rows['final_energy'].tolist()
        combined['electricity_capacity'][ssp] = ssp_rows['electricity_capacity'].tolist()

    return combined

def create_comparison_chart(metric_name, data_dict, ylabel, title, format_str=None):
    """Create a comparison line chart across all SSPs"""
    fig = go.Figure()
    
    for ssp, values in data_dict.items():
        fig.add_trace(go.Scatter(
            x=AI_DATA['years'],
            y=values,
            name=SCENARIOS[ssp]['name'],
            line=dict(color=SCENARIOS[ssp]['color'], width=3),
            mode='lines+markers',
            marker=dict(size=10),
            hovertemplate=f'<b>{SCENARIOS[ssp]["name"]}</b><br>Year: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>'
        ))
    
    # Determine max for dynamic upper bound
    max_val = 0
    for v in data_dict.values():
        try:
            candidate = max(v)
            if candidate > max_val:
                max_val = candidate
        except Exception:
            pass
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#2c3e50')),
        xaxis_title="Year",
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        yaxis=dict(range=[0, max_val * 1.10])
    )
    
    return fig

def create_safety_bar_chart():
    """Create a bar chart showing scalar safety research % for each SSP."""
    ssp_labels = []
    values = []
    colors = []
    for ssp in ['SSP1','SSP2','SSP3','SSP4','SSP5']:
        ssp_labels.append(SCENARIOS[ssp]['name'].split(':')[0])
        safety_value = AI_PROJECTIONS['safety_research_pct'][ssp]
        values.append(safety_value)
        colors.append(SCENARIOS[ssp]['color'])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ssp_labels,
        y=values,
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title=dict(text='Safety Research Investment Share (Representative Level)', font=dict(size=18, color='#2c3e50')),
        xaxis_title='Scenario',
        yaxis_title='Safety Research (%)',
        template='plotly_white',
        height=500,
        yaxis=dict(range=[0, max(values)*1.15])
    )
    return fig

def create_risk_heatmap():
    """Create heatmap of risk probabilities"""
    # Prepare data
    # Desired visual order: SSP1 at top -> SSP5 at bottom
    ordered_ssps = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    ssp_names = [SCENARIOS[ssp]['name'].split(':')[0] for ssp in ordered_ssps]
    z_data = [RISK_DATA[ssp] for ssp in ordered_ssps]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=RISK_DATA['risks'],
        y=ssp_names,
        colorscale='Reds',
        text=z_data,
        texttemplate='%{text}%',
        textfont={"size": 14},
        colorbar=dict(title="Probability (%)")
    ))
    
    fig.update_layout(
        title="Risk Assessment by Scenario (Probability by 2040)",
        xaxis_title="Risk Category",
        yaxis_title="Scenario",
        height=400,
        template='plotly_white',
        yaxis=dict(autorange='reversed')  # Ensure first list element (SSP1) is rendered at top
    )
    
    return fig

def create_radar_chart(ssp):
    """Create radar chart for a specific SSP showing the 5 core AI + systemic risks."""
    categories = RISK_DATA['risks']
    values = RISK_DATA[ssp]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=SCENARIOS[ssp]['color'],
        opacity=0.4,
        line=dict(color=SCENARIOS[ssp]['color'], width=3),
        name=SCENARIOS[ssp]['name']
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        height=450,
        title=f"{SCENARIOS[ssp]['name']}"
    )
    return fig

# Load data
owid_data = load_owid_data()
AI_DATA = get_combined_data(owid_data)  # Use combined OWID + AI projections

# Main unified view
st.header("AI Development Across Shared Socioeconomic Pathways")

st.markdown("---")

# Section 1: OWID Metrics (GDP per capita, Population, CO2)
st.subheader("IPCC Baseline Metric (2020-2040)")

available_metrics = {}
if AI_DATA['gdp_per_capita']:
    available_metrics['GDP per Capita ($/person)'] = ('gdp_per_capita', AI_DATA['gdp_per_capita'])
if AI_DATA['population']:
    available_metrics['Population (Millions)'] = ('population', AI_DATA['population'])
if AI_DATA['co2_emissions']:
    available_metrics['CO₂ Emissions (Gt/year)'] = ('co2_emissions', AI_DATA['co2_emissions'])
if AI_DATA['final_energy']:
    available_metrics['Final Energy (TWh/year)'] = ('final_energy', AI_DATA['final_energy'])
if AI_DATA['electricity_capacity']:
    available_metrics['Electricity Capacity (GW)'] = ('electricity_capacity', AI_DATA['electricity_capacity'])

if not available_metrics:
    st.info("No baseline metrics available from source data.")
else:
    selected_label = st.radio("Select metric", list(available_metrics.keys()), horizontal=True)
    metric_key, metric_data = available_metrics[selected_label]
    fig_metric = create_comparison_chart(metric_key, metric_data, selected_label, f"{selected_label} Trajectories")
    st.plotly_chart(fig_metric, use_container_width=True)

st.markdown("---")

# Section 2: AI-Specific Metrics
st.subheader("AI Development Metrics (2020-2040)")

# Safety research - single chart, centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    fig_safety = create_safety_bar_chart()
    st.plotly_chart(fig_safety, use_container_width=True)

st.markdown("---")

# Section 3: Risk Assessment
st.subheader("Risk Profiles by 2040")

# Risk radar charts - tabs for each SSP
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"{SCENARIOS['SSP1']['name'].split(':')[0]}",
    f"{SCENARIOS['SSP2']['name'].split(':')[0]}",
    f"{SCENARIOS['SSP3']['name'].split(':')[0]}",
    f"{SCENARIOS['SSP4']['name'].split(':')[0]}",
    f"{SCENARIOS['SSP5']['name'].split(':')[0]}"
])

for tab, ssp in zip([tab1, tab2, tab3, tab4, tab5], ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']):
    with tab:
        fig_radar = create_radar_chart(ssp)
        fig_radar.update_layout(height=500)
        st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# Risk heatmap
fig_risk = create_risk_heatmap()
st.plotly_chart(fig_risk, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>AI-SSP Dashboard</strong> | Illustrative proof-of-concept</p>
</div>
""", unsafe_allow_html=True)

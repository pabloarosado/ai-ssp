"""
AI-SSP dashboard: interactive visualization tool of AI safety and governance scenarios based on IPCC Shared Socioeconomic Pathways.
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

# AI Development Metrics (Global) - 5-point scale assessment
# Scale: 1 (lowest/worst) to 5 (highest/best)
AI_METRICS = {
    'SSP1': {
        'Safety': 4,
        'Cooperation': 4,
        'Capability': 3,
    },
    'SSP2': {
        'Safety': 2,
        'Cooperation': 2,
        'Capability': 3,
    },
    'SSP3': {
        'Safety': 1,
        'Cooperation': 1,
        'Capability': 2,
    },
    'SSP4': {
        'Safety': 2,
        'Cooperation': 1,
        'Capability': 3,
    },
    'SSP5': {
        'Safety': 1,
        'Cooperation': 2,
        'Capability': 5,
    }
}

# Risk probabilities by 2040
RISK_DATA = {
    'risks': [
        'Economic\nCollapse',
        'Power\nConflict',
        'Totalitarian\nLock-in',
        'CBRN\nMisuse',
        'Existential\nCatastrophe'
    ],
    'SSP1': [5, 1, 0.5, 1, 0.5],
    'SSP2': [15, 5, 3, 5, 3],
    'SSP3': [15, 10, 7, 10, 10],
    'SSP4': [20, 10, 7, 5, 5],
    'SSP5': [50, 20, 5, 20, 20]
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
    """Load & clean baseline SSP data (2020, 2030, 2040) from cached raw table with regional support."""
    try:
        tb_raw = fetch_ipcc_table()
        tb = tb_raw.reset_index().rename(columns={"country": "scenario"}).drop_duplicates()
        baseline_scenarios = [
            'SSP1 - Baseline', 'SSP2 - Baseline', 'SSP3 - Baseline',
            'SSP4 - Baseline', 'SSP5 - Baseline'
        ]
        tb_filtered = tb[tb['scenario'].isin(baseline_scenarios) & tb['year'].isin([2020, 2030, 2040])].copy()
        tb_filtered['ssp'] = tb_filtered['scenario'].str.extract(r'(SSP\d)')[0]
        
        # Define columns for all regions
        regions = ['global', 'asia', 'latin_america', 'middle_east_and_africa', 'oecd']
        columns_to_keep = {}
        for region in regions:
            columns_to_keep.update({
                f'gdp_per_capita__region_{region}': f'gdp_per_capita_{region}',
                f'population__region_{region}': f'population_people_{region}',
                f'emissions_co2__region_{region}': f'co2_emissions_tonnes_{region}',
                f'final_energy__region_{region}': f'final_energy_{region}',
                f'elec_capacity__region_{region}': f'electricity_capacity_{region}'
            })
        
        base_cols = ['ssp','year']
        available_cols = [c for c in columns_to_keep if c in tb_filtered.columns]
        tb_clean = tb_filtered[base_cols + available_cols].copy()
        rename_map = {k: v for k, v in columns_to_keep.items() if k in available_cols}
        tb_clean = tb_clean.rename(columns=rename_map)
        
        # Convert population and emissions for all regions
        for region in regions:
            pop_col = f'population_people_{region}'
            co2_col = f'co2_emissions_tonnes_{region}'
            
            if pop_col in tb_clean.columns:
                tb_clean[f'population_million_{region}'] = tb_clean[pop_col] / 1e6
                tb_clean = tb_clean.drop(columns=[pop_col])
            
            if co2_col in tb_clean.columns:
                tb_clean[f'co2_emissions_gt_{region}'] = tb_clean[co2_col] / 1e9
                tb_clean = tb_clean.drop(columns=[co2_col])
        
        return tb_clean
    except Exception as e:
        st.warning(f"Could not load OWID data: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_combined_data(_owid_data: Optional[pd.DataFrame] = None, region: str = 'global'):
    """Strictly merge cleaned OWID baseline data with AI projections; no synthetic fallbacks.
    
    Args:
        _owid_data: DataFrame with regional OWID data
        region: Region to extract data for (global, asia, latin_america, middle_east_and_africa, oecd)
    """
    if _owid_data is None:
        raise ValueError("OWID IPCC data not loaded; cannot build baseline metrics.")

    # Build column names for the selected region
    region_suffix = f'_{region}' if region != 'global' else '_global'
    required_cols = ['ssp', 'year',
                     f'gdp_per_capita{region_suffix}',
                     f'population_million{region_suffix}',
                     f'co2_emissions_gt{region_suffix}',
                     f'final_energy{region_suffix}',
                     f'electricity_capacity{region_suffix}']
    
    missing_regional = [c for c in required_cols if c not in _owid_data.columns]
    if missing_regional:
        raise KeyError(f"Missing required columns for region '{region}' in OWID dataset: {missing_regional}")

    years_sorted = sorted(list(set(_owid_data['year'])))
    combined = {
        'years': years_sorted,
        'gdp_per_capita': {},
        'population': {},
        'co2_emissions': {},
        'final_energy': {},
        'electricity_capacity': {}
    }

    for ssp in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
        ssp_rows = _owid_data[_owid_data['ssp'] == ssp].sort_values('year')
        if ssp_rows.empty:
            raise ValueError(f"No rows for {ssp} in OWID data.")
        combined['gdp_per_capita'][ssp] = ssp_rows[f'gdp_per_capita{region_suffix}'].tolist()
        combined['population'][ssp] = ssp_rows[f'population_million{region_suffix}'].tolist()
        combined['co2_emissions'][ssp] = ssp_rows[f'co2_emissions_gt{region_suffix}'].tolist()
        combined['final_energy'][ssp] = ssp_rows[f'final_energy{region_suffix}'].tolist()
        combined['electricity_capacity'][ssp] = ssp_rows[f'electricity_capacity{region_suffix}'].tolist()

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

def create_ai_metrics_table():
    """Create a rating table for AI development metrics across SSPs."""
    import plotly.graph_objects as go
    
    # Define SSPs in order
    ssps = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    metrics = ['Capability', 'Safety', 'Cooperation']
    
    # Create circle rating display (●○○○○ style)
    def create_rating_circles(rating):
        filled = '●' * rating
        empty = '○' * (5 - rating)
        return filled + empty
    
    # Build table data
    ssp_names = [SCENARIOS[ssp]['name'] for ssp in ssps]
    capability_ratings = [create_rating_circles(AI_METRICS[ssp]['Capability']) for ssp in ssps]
    safety_ratings = [create_rating_circles(AI_METRICS[ssp]['Safety']) for ssp in ssps]
    cooperation_ratings = [create_rating_circles(AI_METRICS[ssp]['Cooperation']) for ssp in ssps]
    
    # Create color coding for rows
    row_colors = [SCENARIOS[ssp]['color'] for ssp in ssps]
    # Lighten colors for better readability
    header_color = '#34495e'
    cell_colors = [[f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.15)'] * 3 for c in row_colors]
    
    # Create fill colors for all columns (scenario name column + 3 metric columns)
    # Each row gets the same lightened color across all columns
    fill_colors = []
    for i in range(4):  # 4 columns total
        column_colors = [cell_colors[j][0] for j in range(len(ssps))]
        fill_colors.append(column_colors)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Scenario</b>', '<b>Capability</b>', '<b>Safety</b>', '<b>Cooperation</b>'],
            fill_color=header_color,
            align='left',
            font=dict(color='white', size=14, family='Arial'),
            height=40
        ),
        cells=dict(
            values=[ssp_names, capability_ratings, safety_ratings, cooperation_ratings],
            fill_color=fill_colors,
            align='left',
            font=dict(size=16, family='Arial'),
            height=45
        )
    )])
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
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
        title="Risk assessment by scenario (probability by 2040)",
        # xaxis_title="Risk category",
        yaxis_title="Scenario",
        height=400,
        template='plotly_white',
        yaxis=dict(autorange='reversed'),  # Ensure first list element (SSP1) is rendered at top
        xaxis=dict(side='top')  # Move x-axis labels to top
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
            radialaxis=dict(visible=True, range=[0, 50], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        height=450,
        title=f"{SCENARIOS[ssp]['name']}"
    )
    return fig

# Load data
owid_data = load_owid_data()

# Main unified view
st.header("AI Safety and Governance Across Shared Socioeconomic Pathways")

st.markdown("---")

# Section 1: OWID Metrics (GDP per capita, Population, CO2)
st.subheader("IPCC baseline metrics (2020-2040)")

# Add region selector
REGION_MAPPING = {
    'Global': 'global',
    'Asia': 'asia',
    'Latin America': 'latin_america',
    'Middle East and Africa': 'middle_east_and_africa',
    'OECD': 'oecd'
}

# Load data for selected region (placeholder, will be updated after selection)
# Create elegant layout with metric selector on left and region selector on right
col_metric, col_region = st.columns([3, 1])

# Need to get available metrics first
AI_DATA_temp = get_combined_data(owid_data, region='global')
available_metrics = {}
if AI_DATA_temp['gdp_per_capita']:
    available_metrics['GDP per Capita ($/person)'] = 'gdp_per_capita'
if AI_DATA_temp['population']:
    available_metrics['Population (Millions)'] = 'population'
if AI_DATA_temp['co2_emissions']:
    available_metrics['CO₂ Emissions (Gt/year)'] = 'co2_emissions'
if AI_DATA_temp['final_energy']:
    available_metrics['Final Energy (TWh/year)'] = 'final_energy'
if AI_DATA_temp['electricity_capacity']:
    available_metrics['Electricity Capacity (GW)'] = 'electricity_capacity'

if not available_metrics:
    st.info("No baseline metrics available from source data.")
else:
    with col_metric:
        selected_label = st.radio("Metric", list(available_metrics.keys()), horizontal=True, label_visibility="visible")
    
    with col_region:
        selected_region_label = st.selectbox(
            "Region",
            list(REGION_MAPPING.keys()),
            index=0,  # Default to Global
            label_visibility="visible"
        )
    
    selected_region = REGION_MAPPING[selected_region_label]
    
    # Load data for selected region
    AI_DATA = get_combined_data(owid_data, region=selected_region)
    metric_key = available_metrics[selected_label]
    metric_data = AI_DATA[metric_key]
    
    chart_title = f"{selected_label} Trajectories — {selected_region_label}"
    fig_metric = create_comparison_chart(metric_key, metric_data, selected_label, chart_title)
    st.plotly_chart(fig_metric, use_container_width=True)

st.markdown("---")

# Section 2: AI-Specific Metrics
st.subheader("AI development metrics")

# Display AI metrics table
fig_ai_table = create_ai_metrics_table()
st.plotly_chart(fig_ai_table, use_container_width=True)

st.markdown("---")

# Section 3: Risk Assessment
st.subheader("Risk profiles by 2040")

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
    <p><strong>AI-SSP Dashboard</strong></p>
</div>
""", unsafe_allow_html=True)

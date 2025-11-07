"""Streamlit interface for the fleet transition toolkit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ai, calculators, constants, data, models, report


st.set_page_config(page_title="Fleet Transition Toolkit", layout="wide")


def _init_session_state() -> None:
    if "scenarios" not in st.session_state:
        st.session_state["scenarios"]: List[models.FleetScenario] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def _apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class^="st"]  {
                font-family: 'Inter', sans-serif;
                background-color: #f5f7fb;
            }

            .main-title {
                font-size: 48px;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .main-title span {
                background: linear-gradient(90deg, #00B3FF 0%, #0CCE6B 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-subtitle {
                font-size: 18px;
                color: #4c5464;
                max-width: 760px;
                margin-bottom: 1.5rem;
            }

            .cta-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.75rem 1.8rem;
                border-radius: 14px;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
                border: none;
            }

            .cta-primary {
                background: linear-gradient(135deg, #008CFF 0%, #00D4A3 100%);
                color: white;
                margin-right: 0.75rem;
            }

            .cta-secondary {
                background: #ffffff;
                border: 1px solid #d7deed;
                color: #2c3a4b;
            }

            .feature-card {
                background: white;
                padding: 1.5rem;
                border-radius: 18px;
                box-shadow: 0 10px 40px rgba(15, 30, 65, 0.06);
                height: 100%;
            }

            .feature-card h4 {
                margin-bottom: 0.25rem;
                font-weight: 600;
            }

            .feature-card p {
                color: #596275;
                font-size: 15px;
            }

            .dashboard-section {
                padding: 1.25rem 1.5rem;
                background: white;
                border-radius: 20px;
                box-shadow: 0 15px 45px rgba(15, 30, 65, 0.08);
                margin-bottom: 1.5rem;
            }

            .metric-card {
                background: #f8fbff;
                padding: 1rem 1.25rem;
                border-radius: 16px;
                border: 1px solid #e2e8f3;
            }

            .metric-card .big-number {
                font-size: 26px;
                font-weight: 600;
                margin-bottom: 0.25rem;
            }

            .metric-card .delta {
                font-size: 14px;
                color: #18a558;
            }

            .metric-card .subtitle {
                color: #73819c;
                font-size: 14px;
            }

            .scenario-form .stNumberInput > label,
            .scenario-form .stSelectbox > label,
            .scenario-form .stTextInput > label {
                font-weight: 600;
            }

            .scenario-form .stSlider {
                padding-bottom: 0.5rem;
            }

            .scenario-card {
                background: white;
                padding: 1rem 1.25rem;
                border-radius: 14px;
                border: 1px solid #edf1fa;
            }

            .scenario-card h4 {
                margin: 0;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 0.75rem 1.5rem;
                border-radius: 12px;
                background: rgba(255,255,255,0.9);
                color: #31405a;
                font-weight: 600;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.4rem 0.8rem;
                background: rgba(0, 140, 255, 0.1);
                color: #0369a1;
                border-radius: 999px;
                font-weight: 600;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value: float, unit: str = "", precision: int = 0) -> str:
    return f"{value:,.{precision}f}{unit}"


def _render_home() -> None:
    st.markdown("""<p class="hero-badge">⚡ Fleet Electrification Analysis Platform</p>""", unsafe_allow_html=True)
    st.markdown(
        """
        <h1 class="main-title">Transform Your Fleet with <span>Data-Driven Insights</span></h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="hero-subtitle">
        Advanced modelling and AI-powered analysis to optimise your fleet transition to clean energy. Quantify costs, emissions, and energy demand with precision.
        </p>
        """,
        unsafe_allow_html=True,
    )

    col_cta1, col_cta2 = st.columns([1, 1])
    with col_cta1:
        st.markdown(
            """
            <div class="cta-button cta-primary">View Dashboard</div>
            """,
            unsafe_allow_html=True,
        )
    with col_cta2:
        st.markdown(
            """
            <div class="cta-button cta-secondary">Build Scenario</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("""<div style="height: 1.5rem;"></div>""", unsafe_allow_html=True)

    feature_cols = st.columns(3)
    features = [
        (
            "Economic Analysis",
            "Comprehensive CAPEX/OPEX modelling with TCO insights and sensitivity analysis.",
        ),
        (
            "Emissions Tracking",
            "Well-to-wheel emissions modelling with abatement scenarios and intensity metrics.",
        ),
        (
            "AI Assistant",
            "Natural-language guidance to explore scenarios, explain trade-offs, and draft reports.",
        ),
    ]
    for col, feature in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h4>{feature[0]}</h4>
                    <p>{feature[1]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _generate_dashboard_metrics(analysis: models.ScenarioAnalysis) -> None:
    scenario = analysis.scenario
    total_energy_gwh = analysis.energy.total_kwh / 1_000_000
    total_cost = analysis.economics.tco_per_km * scenario.total_distance_km()
    baseline_emissions = constants.BASELINE_INTENSITY[scenario.vehicle_type] * scenario.total_distance_km()
    abatement_ratio = (
        analysis.environment.abatement_vs_baseline_kg / baseline_emissions if baseline_emissions else 0.0
    )
    total_energy = analysis.energy.total_kwh
    liquid_energy = analysis.energy.energy_by_source_kwh.get(models.EnergySource.LIQUID, 0.0)
    electrified_share = 1.0 - (liquid_energy / total_energy if total_energy else 0.0)
    electrified_share = float(np.clip(electrified_share, 0.0, 1.0))

    metric_cols = st.columns(4)
    metric_cols[0].markdown(
        f"""
        <div class="metric-card">
            <div class="subtitle">Total Energy Demand</div>
            <div class="big-number">{total_energy_gwh:,.0f} GWh</div>
            <div class="delta">Based on scenario duty cycle</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        f"""
        <div class="metric-card">
            <div class="subtitle">Total Cost of Ownership</div>
            <div class="big-number">${total_cost/1_000_000:,.1f}M</div>
            <div class="delta">Amortised 10-year view</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[2].markdown(
        f"""
        <div class="metric-card">
            <div class="subtitle">CO₂ Reduction</div>
            <div class="big-number">{abatement_ratio*100:,.0f}%</div>
            <div class="delta">vs baseline diesel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[3].markdown(
        f"""
        <div class="metric-card">
            <div class="subtitle">Fleet Electrification</div>
            <div class="big-number">{electrified_share*100:,.0f}%</div>
            <div class="delta">Share of non-liquid energy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard_charts(analysis: models.ScenarioAnalysis) -> None:
    years = np.array([2025, 2030, 2035, 2040])
    electric_base = (
        analysis.energy.energy_by_source_kwh.get(models.EnergySource.GRID, 0.0)
        + analysis.energy.energy_by_source_kwh.get(models.EnergySource.RENEWABLE, 0.0)
    )
    hydro_base = analysis.energy.energy_by_source_kwh.get(models.EnergySource.HYDROGEN, 0.0)
    liquid_base = analysis.energy.energy_by_source_kwh.get(models.EnergySource.LIQUID, 0.0)

    energy_projection = pd.DataFrame(
        {
            "Year": years,
            "Electricity (GWh)": np.linspace(electric_base * 0.6, electric_base * 1.4, len(years)) / 1_000_000,
            "Hydrogen (GWh)": np.linspace(hydro_base * 0.5, hydro_base * 1.3, len(years)) / 1_000_000,
            "Diesel (GWh)": np.linspace(liquid_base, liquid_base * 0.1, len(years)) / 1_000_000,
        }
    )

    energy_chart = px.bar(
        energy_projection,
        x="Year",
        y=["Electricity (GWh)", "Hydrogen (GWh)", "Diesel (GWh)"],
        barmode="group",
        color_discrete_sequence=["#0074FF", "#22C58B", "#8891A5"],
    )
    energy_chart.update_layout(margin=dict(t=10, l=10, r=10, b=10))

    cost_df = pd.DataFrame(
        {
            "Year": years,
            "CAPEX ($M)": np.linspace(analysis.economics.capex_total * 1.1, analysis.economics.capex_total * 0.8, len(years))
            / 1_000_000,
            "OPEX ($M)": np.linspace(analysis.economics.opex_total * 1.05, analysis.economics.opex_total * 0.85, len(years))
            / 1_000_000,
            "TCO ($M)": np.linspace(
                analysis.economics.tco_per_km * analysis.scenario.total_distance_km() * 1.05,
                analysis.economics.tco_per_km * analysis.scenario.total_distance_km() * 0.8,
                len(years),
            )
            / 1_000_000,
        }
    )
    cost_chart = px.line(
        cost_df,
        x="Year",
        y=["CAPEX ($M)", "OPEX ($M)", "TCO ($M)"],
        color_discrete_sequence=["#ff9f1a", "#3d55f9", "#12b886"],
    )
    cost_chart.update_layout(margin=dict(t=10, l=10, r=10, b=10))

    baseline_intensity = constants.BASELINE_INTENSITY[analysis.scenario.vehicle_type]
    intensity_df = pd.DataFrame(
        {
            "Year": years,
            "Baseline (Diesel)": np.full(len(years), baseline_intensity * 1000),
            "Electrification Scenario": np.linspace(
                baseline_intensity * 1000 * 0.95,
                analysis.environment.intensity_per_km * 1000,
                len(years),
            ),
            "Target": np.linspace(baseline_intensity * 1000 * 0.9, baseline_intensity * 1000 * 0.2, len(years)),
        }
    )
    intensity_chart = px.line(
        intensity_df,
        x="Year",
        y=["Baseline (Diesel)", "Electrification Scenario", "Target"],
        color_discrete_sequence=["#9ca3af", "#0284c7", "#34d399"],
    )
    intensity_chart.update_layout(margin=dict(t=10, l=10, r=10, b=10))

    total_abatement = max(analysis.environment.abatement_vs_baseline_kg, 1.0)
    abatement_df = pd.DataFrame(
        {
            "Year": years,
            "Diesel Reduction": np.linspace(total_abatement * 0.6, total_abatement * 0.05, len(years)) / 1_000,
            "Electric Savings": np.linspace(total_abatement * 0.2, total_abatement * 0.4, len(years)) / 1_000,
            "Hydrogen Savings": np.linspace(total_abatement * 0.1, total_abatement * 0.25, len(years)) / 1_000,
            "Renewable Savings": np.linspace(total_abatement * 0.05, total_abatement * 0.3, len(years)) / 1_000,
        }
    )
    abatement_chart = px.area(
        abatement_df,
        x="Year",
        y=["Diesel Reduction", "Electric Savings", "Hydrogen Savings", "Renewable Savings"],
        color_discrete_sequence=["#9ca3af", "#0ea5e9", "#6366f1", "#f59e0b"],
    )
    abatement_chart.update_layout(margin=dict(t=10, l=10, r=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02))

    top_row = st.columns(2)
    with top_row[0]:
        st.markdown("### Energy Mix Projection")
        st.plotly_chart(energy_chart, use_container_width=True)
    with top_row[1]:
        st.markdown("### Cost Analysis")
        st.plotly_chart(cost_chart, use_container_width=True)

    bottom_row = st.columns(2)
    with bottom_row[0]:
        st.markdown("### Emissions Intensity")
        st.plotly_chart(intensity_chart, use_container_width=True)
    with bottom_row[1]:
        st.markdown("### Cumulative Abatement")
        st.plotly_chart(abatement_chart, use_container_width=True)


def _render_dashboard(analyses: List[models.ScenarioAnalysis]) -> None:
    if not analyses:
        st.info("Add a scenario in the Scenarios tab to unlock the dashboard.")
        return

    analysis = analyses[0]
    st.markdown("## Fleet Analysis Dashboard")
    st.caption("Real-time insights into your fleet transition scenario.")

    _generate_dashboard_metrics(analysis)
    st.markdown("""<div style="height: 1rem;"></div>""", unsafe_allow_html=True)
    _render_dashboard_charts(analysis)


def _render_scenario_builder() -> None:
    st.markdown("## Scenario Builder")
    st.caption("Configure fleet parameters and run economic and emissions analyses.")

    with st.form("scenario-form", clear_on_submit=False):
        st.markdown("""<div class="scenario-form">""", unsafe_allow_html=True)
        left_col, right_col = st.columns(2)
        with left_col:
            vehicle_type = st.selectbox("Vehicle Type", constants.VEHICLE_TYPES, key="vehicle_type")
            quantity = int(st.number_input("Fleet Size", min_value=1, value=50, step=1, key="quantity"))
            mileage = st.number_input(
                "Annual Mileage (km/vehicle)", min_value=5_000, value=40_000, step=1_000, key="mileage"
            )
            duty_cycle = st.number_input(
                "Duty Cycle (hours/day)", min_value=1.0, max_value=24.0, value=8.0, step=0.5, key="duty_cycle"
            )
            depot = st.text_input("Depot Location", value="London", key="depot_location")

        with right_col:
            grid_share = st.slider("Grid Electricity (%)", 0, 100, 40, step=5, key="grid_share")
            renewable_share = st.slider("Renewable Energy (%)", 0, 100, 30, step=5, key="renewable_share")
            hydrogen_share = st.slider("Hydrogen (%)", 0, 100, 20, step=5, key="hydrogen_share")
            liquid_share = st.slider("Liquid Fuels (%)", 0, 100, 10, step=5, key="liquid_share")
            notes = st.text_area("Scenario Notes", value="", key="scenario_notes")

        st.markdown("""</div>""", unsafe_allow_html=True)

        submitted = st.form_submit_button("Run Scenario", use_container_width=True)

    if submitted:
        if grid_share + renewable_share + hydrogen_share + liquid_share == 0:
            st.warning("Please allocate at least one energy source.")
        else:
            scenario = models.FleetScenario(
                name=f"{vehicle_type} Fleet {quantity}",
                vehicle_type=vehicle_type,
                quantity=quantity,
                annual_mileage_km=mileage,
                duty_cycle_hours_per_day=duty_cycle,
                depot_location=depot.strip() or "Unknown",
                energy_mix=models.EnergyMix(
                    grid_share=grid_share,
                    renewable_share=renewable_share,
                    hydrogen_share=hydrogen_share,
                    liquid_share=liquid_share,
                ),
                notes=notes.strip() or None,
            )

            existing_names = [s.name for s in st.session_state["scenarios"]]
            if scenario.name in existing_names:
                idx = existing_names.index(scenario.name)
                st.session_state["scenarios"][idx] = scenario
                st.success(f"Updated scenario '{scenario.name}'.")
            else:
                st.session_state["scenarios"].append(scenario)
                st.success(f"Added scenario '{scenario.name}'.")

            st.experimental_rerun()

    if st.session_state["scenarios"]:
        st.markdown("### Saved Scenarios")
        for scenario in st.session_state["scenarios"]:
            with st.container():
                st.markdown(
                    f"""
                    <div class="scenario-card">
                        <h4>{scenario.name}</h4>
                        <p style="color:#6b7488;">{scenario.quantity} {scenario.vehicle_type}(s) • {scenario.annual_mileage_km:,.0f} km/year • {scenario.depot_location}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("Explore Source Datasets"):
        datasets = data.list_available_datasets()
        dataset_label = st.selectbox("Choose dataset", list(datasets), index=0, key="dataset_picker")
        df = data.load_dataset(dataset_label)
        st.caption(
            f"Loaded {datasets[dataset_label].name} ({len(df):,} rows, {len(df.columns)} columns)"
        )
        st.dataframe(df.head(200), use_container_width=True)


def _render_ai_assistant(analyses: List[models.ScenarioAnalysis]) -> None:
    st.markdown("## AI Assistant")
    assistant = ai.FleetAssistant()
    context = models.summarise_scenarios(analyses)
    st.caption("Ask FleetGuide for scenario ideas, explainers, or trade-offs.")

    with st.form("ai-form"):
        prompt = st.text_area("Your question", height=160)
        submitted = st.form_submit_button("Ask FleetGuide")

    if submitted:
        answer = assistant.generate_response(prompt, context=context)
        st.session_state["chat_history"].append({"question": prompt, "answer": answer})

    if not assistant.available:
        st.warning(
            "No OpenAI API key detected. Set the OPENAI_API_KEY environment variable to enable AI responses."
        )

    for item in reversed(st.session_state["chat_history"]):
        st.write("**You:**", item["question"])
        st.write("**FleetGuide:**", item["answer"])
        st.write("---")


def _render_export_section(analyses: List[models.ScenarioAnalysis]) -> None:
    if not analyses:
        st.info("Run a scenario first to export results.")
        return

    st.markdown("## Export Report")
    pdf_bytes = report.build_pdf_report(analyses)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="fleet-transition-report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.code(json.dumps([analysis.to_dict() for analysis in analyses], indent=2))


def main() -> None:
    _init_session_state()
    _apply_custom_styles()

    home_tab, dashboard_tab, scenarios_tab, ai_tab, export_tab = st.tabs(
        ["Home", "Dashboard", "Scenarios", "AI Assistant", "Export"]
    )

    with home_tab:
        _render_home()

    analyses = [calculators.run_full_analysis(scenario) for scenario in st.session_state["scenarios"]]

    with dashboard_tab:
        _render_dashboard(analyses)

    with scenarios_tab:
        _render_scenario_builder()

    with ai_tab:
        _render_ai_assistant(analyses)

    with export_tab:
        _render_export_section(analyses)


if __name__ == "__main__":
    main()


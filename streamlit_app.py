"""Streamlit interface for the fleet transition toolkit."""

from __future__ import annotations

import json
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from app import ai, calculators, constants, data, models, report


st.set_page_config(page_title="Fleet Transition Toolkit", layout="wide")


def _init_session_state() -> None:
    if "scenarios" not in st.session_state:
        st.session_state["scenarios"]: List[models.FleetScenario] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def _energy_mix_controls() -> models.EnergyMix:
    st.subheader("Energy mix")
    grid_share = st.slider("Grid electricity", 0, 100, 40, step=5)
    renewable_share = st.slider("Renewable electricity", 0, 100, 30, step=5)
    hydrogen_share = st.slider("Green hydrogen", 0, 100, 20, step=5)
    liquid_share = st.slider("Liquid fuels", 0, 100, 10, step=5)

    if grid_share + renewable_share + hydrogen_share + liquid_share == 0:
        st.warning("Set at least one energy source above zero.")

    return models.EnergyMix(
        grid_share=grid_share,
        renewable_share=renewable_share,
        hydrogen_share=hydrogen_share,
        liquid_share=liquid_share,
    )


def _scenario_form() -> None:
    st.sidebar.header("Scenario builder")

    name = st.sidebar.text_input("Scenario name", value="Example Fleet")
    vehicle_type = st.sidebar.selectbox("Vehicle type", constants.VEHICLE_TYPES)
    quantity = int(st.sidebar.number_input("Quantity", min_value=1, value=25))
    mileage = st.sidebar.number_input(
        "Annual mileage per vehicle (km)", min_value=1_000, value=40_000, step=1_000
    )
    duty_cycle = st.sidebar.number_input(
        "Duty cycle (hours/day)", min_value=1.0, max_value=24.0, value=8.0, step=0.5
    )
    depot = st.sidebar.text_input("Depot / geography", value="London")
    notes = st.sidebar.text_area("Notes", value="")

    with st.sidebar.expander("Energy mix", expanded=True):
        mix = _energy_mix_controls()

    if st.sidebar.button("Add / update scenario", use_container_width=True):
        scenario = models.FleetScenario(
            name=name.strip() or "Untitled scenario",
            vehicle_type=vehicle_type,
            quantity=quantity,
            annual_mileage_km=mileage,
            duty_cycle_hours_per_day=duty_cycle,
            depot_location=depot.strip() or "Unknown",
            energy_mix=mix,
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

    if st.sidebar.button("Clear all scenarios", use_container_width=True):
        st.session_state["scenarios"].clear()
        st.sidebar.info("All scenarios removed.")


def _render_dataset_browser() -> None:
    st.header("Dataset explorer")
    datasets = data.list_available_datasets()
    dataset_label = st.selectbox("Choose dataset", list(datasets), index=0)
    df = data.load_dataset(dataset_label)

    st.caption(f"Loaded {datasets[dataset_label].name} ({len(df):,} rows, {len(df.columns)} columns)")
    st.dataframe(df.head(200), use_container_width=True)

    with st.expander("Plot time series (long format)"):
        tidy_df = data.melt_time_series(df)
        st.dataframe(tidy_df.head(200), use_container_width=True)
        if not tidy_df.empty:
            chart = px.line(tidy_df, x="Year", y="Value", color=tidy_df.columns[0])
            st.plotly_chart(chart, use_container_width=True)


def _render_scenario_results(analyses: List[models.ScenarioAnalysis]) -> None:
    if not analyses:
        st.info("Add a scenario in the sidebar to see analytics.")
        return

    st.header("Scenario analytics")

    for analysis in analyses:
        st.subheader(analysis.scenario.name)

        cols = st.columns(4)
        cols[0].metric("Total energy", f"{analysis.energy.total_kwh:,.0f} kWh/yr")
        cols[1].metric("Hydrogen", f"{analysis.energy.hydrogen_kg:,.1f} kg/yr")
        cols[2].metric("TCO per km", f"${analysis.economics.tco_per_km:,.2f}")
        cols[3].metric("Emissions", f"{analysis.environment.intensity_per_km:,.3f} kg/km")

        source_order = [
            models.EnergySource.GRID,
            models.EnergySource.RENEWABLE,
            models.EnergySource.HYDROGEN,
            models.EnergySource.LIQUID,
        ]
        mix_df = pd.DataFrame(
            {
                "Energy source": [source.value for source in source_order],
                "kWh": [analysis.energy.energy_by_source_kwh.get(source, 0.0) for source in source_order],
            }
        )
        st.plotly_chart(px.pie(mix_df, names="Energy source", values="kWh"), use_container_width=True)

        with st.expander("Detailed outputs", expanded=False):
            st.write("**Energy breakdown (kWh)**")
            energy_table = {
                source.value: value for source, value in analysis.energy.energy_by_source_kwh.items()
            }
            energy_table["Hydrogen (kg)"] = analysis.energy.hydrogen_kg
            energy_table["Liquid fuels (L)"] = analysis.energy.total_liquid_litres
            st.json(energy_table)

            st.write("**Economics**")
            st.json(analysis.economics.__dict__)

            st.write("**Environment**")
            st.json(analysis.environment.__dict__)


def _render_ai_assistant(analyses: List[models.ScenarioAnalysis]) -> None:
    st.header("AI assistant")
    assistant = ai.FleetAssistant()
    context = models.summarise_scenarios(analyses)

    st.caption("Ask the assistant for scenario ideas, explainers, or trade-offs.")

    with st.form("ai-form"):
        prompt = st.text_area("Your question", height=120)
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
        return

    st.header("Export")
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
    _scenario_form()

    tabs = st.tabs(["Datasets", "Analytics", "AI assistant", "Export"])

    with tabs[0]:
        _render_dataset_browser()

    analyses = [calculators.run_full_analysis(scenario) for scenario in st.session_state["scenarios"]]

    with tabs[1]:
        _render_scenario_results(analyses)

    with tabs[2]:
        _render_ai_assistant(analyses)

    with tabs[3]:
        _render_export_section(analyses)


if __name__ == "__main__":
    main()


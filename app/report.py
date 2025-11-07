"""PDF report builder for fleet analyses."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List

import matplotlib.pyplot as plt
from fpdf import FPDF

from .models import EnergySource, ScenarioAnalysis


@dataclass
class ReportFigure:
    title: str
    image_bytes: bytes


def _scenario_energy_chart(analysis: ScenarioAnalysis) -> ReportFigure:
    labels = [
        EnergySource.GRID.value,
        EnergySource.RENEWABLE.value,
        EnergySource.HYDROGEN.value,
        EnergySource.LIQUID.value,
    ]
    values = [
        analysis.energy.energy_by_source_kwh.get(EnergySource.GRID, 0.0),
        analysis.energy.energy_by_source_kwh.get(EnergySource.RENEWABLE, 0.0),
        analysis.energy.energy_by_source_kwh.get(EnergySource.HYDROGEN, 0.0),
        analysis.energy.energy_by_source_kwh.get(EnergySource.LIQUID, 0.0),
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color=["#2E86AB", "#1B998B", "#C3423F", "#F0A202"])
    ax.set_ylabel("Energy demand (kWh)")
    ax.set_title(f"Energy mix for {analysis.scenario.name}")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)

    return ReportFigure(title=f"Energy mix - {analysis.scenario.name}", image_bytes=buf.read())


def build_figures(analyses: Iterable[ScenarioAnalysis]) -> List[ReportFigure]:
    figures: List[ReportFigure] = []
    for analysis in analyses:
        figures.append(_scenario_energy_chart(analysis))
    return figures


def build_pdf_report(analyses: Iterable[ScenarioAnalysis]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    analyses_list = list(analyses)
    figures = build_figures(analyses_list)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Fleet Transition Analysis", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        6,
        "This report summarises the energy demand, economic performance, and emissions outcomes "
        "for the configured fleet scenarios.",
    )

    for analysis in analyses_list:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"Scenario: {analysis.scenario.name}", ln=True)

        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0,
            6,
            (
                f"Vehicles: {analysis.scenario.quantity} {analysis.scenario.vehicle_type}(s) based in {analysis.scenario.depot_location}.\n"
                f"Annual mileage per vehicle: {analysis.scenario.annual_mileage_km:,.0f} km.\n"
                f"Duty cycle: {analysis.scenario.duty_cycle_hours_per_day:,.1f} hours/day."
            ),
        )

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Energy demand", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0,
            6,
            (
                f"Total energy: {analysis.energy.total_kwh:,.0f} kWh/year. "
                f"Hydrogen: {analysis.energy.hydrogen_kg:,.1f} kg/year. "
                f"Liquid fuels: {analysis.energy.total_liquid_litres:,.0f} L/year."
            ),
        )

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Economics", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0,
            6,
            (
                f"Capex: ${analysis.economics.capex_total:,.0f}\n"
                f"Opex (annual): ${analysis.economics.opex_total:,.0f}\n"
                f"TCO per km: ${analysis.economics.tco_per_km:,.2f}"
            ),
        )

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Emissions", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0,
            6,
            (
                f"Total emissions: {analysis.environment.total_emissions_kg:,.0f} kg CO2e/year\n"
                f"Intensity: {analysis.environment.intensity_per_km:,.3f} kg/km\n"
                f"Abatement vs. baseline: {analysis.environment.abatement_vs_baseline_kg:,.0f} kg CO2e/year"
            ),
        )

        if analysis.scenario.notes:
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 6, f"Notes: {analysis.scenario.notes}")

        # Embed chart if available
        figure = next((fig for fig in figures if fig.title.endswith(analysis.scenario.name)), None)
        if figure:
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(figure.image_bytes)
                tmp.flush()
                tmp_path = Path(tmp.name)
            pdf.image(str(tmp_path), w=160)
            tmp_path.unlink(missing_ok=True)

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes


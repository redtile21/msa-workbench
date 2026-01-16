import os
import tempfile
from fpdf import FPDF

from msa_workbench.engine.msa_engine import MSAResult
from msa_workbench.plotting import get_variability_chart, get_stddev_chart
from msa_workbench.reporting.analysis_notes import get_variation_impact_analysis


def create_pdf_report(result: MSAResult) -> bytes:
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Gage R&R Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # 1. Summary Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Summary Metrics", 0, 1)
    pdf.set_font("Arial", size=10)

    grr_sv = result.grr_summary.total_gage_rr_pct_study_var
    grr_tol = result.grr_summary.total_gage_rr_pct_tolerance
    ndc = result.grr_summary.ndc

    pdf.cell(60, 10, f"Gage R&R (%SV): {grr_sv:.2f}%", 1)
    pdf.cell(60, 10, f"Gage R&R (%Tol): {grr_tol:.2f}%" if grr_tol else "Gage R&R (%Tol): N/A", 1)
    pdf.cell(60, 10, f"ndc: {ndc:.1f}" if ndc else "ndc: N/A", 1)
    pdf.ln(15)

    pdf.multi_cell(0, 10, f"Interpretation: {result.grr_summary.interpretation}")
    pdf.ln(5)

    # 2. Variance Components
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Variance Components", 0, 1)
    pdf.set_font("Arial", size=8)

    # Header
    cols = ["Source", "Var Comp", "Variability (6*SD)", "% Contrib", "% Study Var", "% Tol"]
    col_widths = [50, 25, 30, 25, 25, 25]
    for i, h in enumerate(cols):
        pdf.cell(col_widths[i], 7, h, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", size=8)
    for r in result.var_components:
        pct_tol_str = f"{r.pct_tolerance:.1f}" if r.pct_tolerance is not None else "N/A"
        pdf.cell(col_widths[0], 6, r.source, 1)
        pdf.cell(col_widths[1], 6, f"{r.var_comp:.4g}", 1, 0, 'R')
        pdf.cell(col_widths[2], 6, f"{r.variability:.4g}", 1, 0, 'R')
        pdf.cell(col_widths[3], 6, f"{r.pct_contribution:.1f}", 1, 0, 'R')
        pdf.cell(col_widths[4], 6, f"{r.pct_study_var:.1f}", 1, 0, 'R')
        pdf.cell(col_widths[5], 6, pct_tol_str, 1, 0, 'R')
        pdf.ln()
    pdf.ln(5)

    # Variation Impact Text
    impacts = get_variation_impact_analysis(result)
    pdf.set_font("Arial", size=9)
    for type_, msg in impacts:
        if type_ == 'success':
            pdf.set_text_color(0, 100, 0)
        elif type_ == 'error':
            pdf.set_text_color(150, 0, 0)
        else:
            pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, msg)
        pdf.ln(1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # 3. ANOVA Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "ANOVA Table", 0, 1)
    pdf.set_font("Arial", size=8)

    # Header
    cols = ["Source", "DF", "SS", "MS", "F", "p-value"]
    col_widths = [60, 15, 30, 30, 20, 20]
    for i, h in enumerate(cols):
        pdf.cell(col_widths[i], 7, h, 1, 0, 'C')
    pdf.ln()

    for r in result.anova_table:
        f_str = f"{r.f:.3f}" if r.f is not None else "N/A"
        p_str = f"{r.p:.4f}" if r.p is not None else "N/A"

        pdf.cell(col_widths[0], 6, r.term, 1)
        pdf.cell(col_widths[1], 6, f"{r.df:.0f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 6, f"{r.ss:.4g}", 1, 0, 'R')
        pdf.cell(col_widths[3], 6, f"{r.ms:.4g}", 1, 0, 'R')
        pdf.cell(col_widths[4], 6, f_str, 1, 0, 'R')
        pdf.cell(col_widths[5], 6, p_str, 1, 0, 'R')
        pdf.ln()
    pdf.ln(5)

    # 4. Charts
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Charts", 0, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Variability Chart
        fig_var = get_variability_chart(result)
        path_var = os.path.join(tmpdir, "var_chart.png")
        fig_var.savefig(path_var, bbox_inches='tight', dpi=100)
        pdf.image(path_var, x=10, w=190)
        pdf.ln(5)

        # Std Dev Chart
        fig_std = get_stddev_chart(result)
        path_std = os.path.join(tmpdir, "std_chart.png")
        fig_std.savefig(path_std, bbox_inches='tight', dpi=100)
        pdf.add_page()
        pdf.image(path_std, x=10, w=190)

    # Fix: Handle return type for different FPDF versions
    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)


def save_pdf_report(result: MSAResult, output_path: str):
    """Generates and saves the PDF report to a file."""
    pdf_bytes = create_pdf_report(result)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

import io
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import matplotlib.pyplot as plt

from msa_workbench.engine.msa_engine import MSAResult
from msa_workbench.plotting import get_variability_chart, get_stddev_chart
from msa_workbench.reporting.analysis_notes import get_variation_impact_analysis

def _figure_to_image_stream(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf

def _format_value(val):
    if isinstance(val, (int, float)):
        return "{:.4g}".format(val)
    return str(val)

def _add_df_to_table(slide, df, left, top, width, height):
    rows, cols = df.shape
    shape = slide.shapes.add_table(rows + 1, cols, left, top, width, height)
    table = shape.table

    # Set column widths: make first column wider
    if cols > 1:
        table.columns[0].width = int(width * 0.4)
        other_col_width = int((width * 0.6) / (cols - 1))
        for i in range(1, cols):
            table.columns[i].width = other_col_width
    else:
        table.columns[0].width = width

    light_gray = RGBColor(211, 211, 211)

    # Add header and shade it
    for col_idx, col_name in enumerate(df.columns):
        cell = table.cell(0, col_idx)
        cell.text = str(col_name)
        run = cell.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(10)
        run.font.bold = True
        cell.fill.solid()
        cell.fill.fore_color.rgb = light_gray

    # Add data and shade the first column
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r + 1, c)
            value = df.iloc[r, c]
            cell.text = _format_value(value)
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(9)
            if c == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = light_gray
            else:
                cell.fill.background() # No fill

    return shape

def save_pptx_report(result: MSAResult, template_path: str, output_path: str):
    """
    Generates a PowerPoint report from the MSA result.
    """
    try:
        prs = Presentation(template_path)
    except Exception as e:
        raise IOError(f"Could not open template file: {template_path}") from e

    if not prs.slides:
        raise ValueError("The PowerPoint template is empty and has no slides.")
    
    slide_layout = prs.slides[0].slide_layout

    while len(prs.slides) < 5:
        prs.slides.add_slide(slide_layout)

    # Slide 1: Summary and Warnings
    slide1 = prs.slides[0]
    if slide1.shapes.title:
        slide1.shapes.title.text = "Gauge R&R Summary"
    
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(5.5)
    txBox = slide1.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.clear()

    summary = result.grr_summary
    
    p1 = tf.paragraphs[0]
    p1.text = f"GRR % Study Var: {summary.total_gage_rr_pct_study_var:.2f}%"
    
    p2 = tf.add_paragraph()
    p2.text = (
        f"GRR % Tolerance: {summary.total_gage_rr_pct_tolerance:.2f}%"
        if summary.total_gage_rr_pct_tolerance is not None
        else "GRR % Tolerance: N/A"
    )
    
    p3 = tf.add_paragraph()
    p3.text = f"NDC: {summary.ndc}"

    p4 = tf.add_paragraph()
    p4.text = f"Interpretation: {summary.interpretation}"
    
    if result.warnings:
        p5 = tf.add_paragraph()
        p5.text = "\nWarnings:"
        for warning in result.warnings:
            p_warn = tf.add_paragraph()
            p_warn.text = f"- {warning}"
            p_warn.level = 1


    # Slide 2: Tables
    slide2 = prs.slides[1]
    if slide2.shapes.title:
        slide2.shapes.title.text = "Variance Components and ANOVA"
    
    var_comp_df = pd.DataFrame(result.var_components)
    _add_df_to_table(slide2, var_comp_df, Inches(0.5), Inches(1.5), Inches(4.25), Inches(4))

    anova_df = pd.DataFrame(result.anova_table)
    _add_df_to_table(slide2, anova_df, Inches(5.25), Inches(1.5), Inches(4.25), Inches(4))

    # Slide 3: Variability Chart
    slide3 = prs.slides[2]
    if slide3.shapes.title:
        slide3.shapes.title.text = "Variability Chart"
    
    fig_var = plt.figure(figsize=(10, 5.5), dpi=300)
    ax_var = fig_var.add_subplot(111)
    get_variability_chart(result, ax_var)
    fig_var.tight_layout(pad=2.0)
    
    image_stream_var = _figure_to_image_stream(fig_var)
    slide3.shapes.add_picture(image_stream_var, Inches(0.5), Inches(1.5), width=Inches(9))
    plt.close(fig_var)
    
    # Slide 4: Std Dev Chart
    slide4 = prs.slides[3]
    if slide4.shapes.title:
        slide4.shapes.title.text = "Standard Deviation Chart"

    fig_std = plt.figure(figsize=(10, 5.5), dpi=300)
    ax_std = fig_std.add_subplot(111)
    get_stddev_chart(result, ax_std)
    fig_std.tight_layout(pad=2.0)
    
    image_stream_std = _figure_to_image_stream(fig_std)
    slide4.shapes.add_picture(image_stream_std, Inches(0.5), Inches(1.5), width=Inches(9))
    plt.close(fig_std)

    # Slide 5: Analysis Notes
    slide5 = prs.slides[4]
    if slide5.shapes.title:
        slide5.shapes.title.text = "Analysis Notes"

    impacts = get_variation_impact_analysis(result)
    
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(5.5)
    txBox = slide5.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.clear()

    for type_, msg in impacts:
        p = tf.add_paragraph()
        p.text = msg
        font = p.runs[0].font
        if type_ == "success":
            font.color.rgb = RGBColor(0x00, 0x80, 0x00) # Green
        elif type_ == "error":
            font.color.rgb = RGBColor(0xFF, 0x00, 0x00) # Red
        else:
            font.color.rgb = RGBColor(0x00, 0x00, 0x00) # Black

    prs.save(output_path)

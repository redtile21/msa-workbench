from msa_workbench.engine.msa_engine import MSAResult


def get_variation_impact_analysis(result: MSAResult) -> list:
    """Returns a list of (type, message) tuples for the impact analysis."""
    impacts = []

    def get_contrib(source_name_exact=None, source_name_part=None):
        for r in result.var_components:
            if source_name_exact and r.source == source_name_exact:
                return r.pct_contribution
            if source_name_part and source_name_part in r.source:
                return r.pct_contribution
        return 0.0

    pct_grr = get_contrib(source_name_exact="Gage R&R")
    pct_repeat = get_contrib(source_name_exact="Repeatability")
    pct_part = get_contrib(source_name_part="Part-to-Part")
    pct_repro = max(0.0, pct_grr - pct_repeat)

    # A. Analyze Measurement System vs. Part Variation
    if pct_part > pct_grr:
        impacts.append(("success",
                        f"Most Impactful Factor: The results are primarily driven by Part-to-Part Variation ({pct_part:.1f}%). "
                        "This is the desired outcome, indicating the measurement system can effectively distinguish between different parts."
                        ))
    else:
        impacts.append(("error",
                        f"Most Impactful Factor: The results are primarily driven by Measurement System Variation (Gage R&R) ({pct_grr:.1f}%). "
                        "This indicates the measurement system is introducing more noise than the actual differences between the parts."
                        ))

    # B. Analyze Repeatability vs. Reproducibility
    if pct_grr > 0.1:
        impacts.append(("info", "Breakdown of Measurement Error:"))
        if pct_repeat > pct_repro:
            impacts.append(("info",
                            f"- Repeatability is the dominant source ({pct_repeat:.1f}% vs {pct_repro:.1f}%). "
                            "Suggests issues with the gage/equipment or method consistency."
                            ))
        else:
            impacts.append(("info",
                            f"- Reproducibility is the dominant source ({pct_repro:.1f}% vs {pct_repeat:.1f}%). "
                            "Suggests issues with operator differences (training, technique)."
                            ))

    return impacts

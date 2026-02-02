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
        impacts.append((
            "success",
            f"Most Impactful Factor: The results are primarily driven by Part-to-Part Variation ({pct_part:.1f}%). "
            "This is the desired outcome, indicating the measurement system can effectively distinguish between different parts."
        ))

        # Lab-focused tips when part variation dominates (measurement system is likely OK)
        impacts.append(("info", "Lab tips to keep results reliable:"))
        impacts.append(("info", "- Verify calibration status and run a check standard/QC sample each run (consider start/middle/end for long sequences)."))
        impacts.append(("info", "- Protect the true part-to-part signal: standardize sample prep, mixing, and dilution paths so prep variability doesn't mask differences."))
        impacts.append(("info", "- Control time effects: define max hold time and storage conditions; watch for drift from warm-up, evaporation, or degradation."))
        impacts.append(("info", "- Randomize run order to prevent slow drift from being mistaken for part differences (especially for plate-based or long analytical runs)."))
        impacts.append(("info", "- Record critical conditions (reagent lots/age, temperatures, incubation windows, method version, analysis settings) to ensure repeatability across days."))

    else:
        impacts.append((
            "error",
            f"Most Impactful Factor: The results are primarily driven by Measurement System Variation (Gage R&R) ({pct_grr:.1f}%). "
            "This indicates the measurement system is introducing more noise than the actual differences between the parts."
        ))

        # Lab-focused tips when gage dominates (measurement system is too noisy)
        impacts.append(("info", "High-impact fixes common in chemistry/biology labs:"))
        impacts.append(("info", "- Calibration & verification: confirm instrument is in calibration; independently verify with a check standard/QC (bracket samples with QC to detect drift)."))
        impacts.append(("info", "- Pipetting technique & range: use the correct volume range (avoid the bottom ~10% of the pipette range), pre-wet tips, and keep aspiration/dispense speed and angle consistent."))
        impacts.append(("info", "- Dilutions: prefer single-step or gravimetric dilutions for critical steps; standardize mixing (vortex speed/time) and avoid unnecessary serial dilutions."))
        impacts.append(("info", "- Work instructions: remove ambiguity (exact units, order of addition, mixing method/time, timing windows, temperatures, acceptance criteria). Replace 'mix well' with specifics."))
        impacts.append(("info", "- Time/evaporation/degradation: define max time from prep→read; cover plates/tubes, control temperature, and protect light/oxygen-sensitive analytes."))
        impacts.append(("info", "- Randomization & blocking: randomize sample order across operators and time; if runs span hours/days, block by day/batch and include QC per block."))

    # B. Analyze Repeatability vs. Reproducibility
    if pct_grr > 0.1:
        impacts.append(("info", "Breakdown of Measurement Error:"))
        if pct_repeat > pct_repro:
            impacts.append((
                "info",
                f"- Repeatability is the dominant source ({pct_repeat:.1f}% vs {pct_repro:.1f}%). "
                "Suggests issues with the gage/equipment or method consistency."
            ))

            # Repeatability troubleshooting tips (within-operator variation)
            impacts.append(("info", "Repeatability troubleshooting (within-operator variation):"))
            impacts.append(("info", "- Instrument stability: allow warm-up/equilibration; check baseline noise, leaks/pressure issues, temperature control, detector saturation, and aging consumables (lamp/laser)."))
            impacts.append(("info", "- Consumables & fixtures: standardize tips/plates/cuvettes/seals and verify alignment/positioning (misalignment often looks like noise)."))
            impacts.append(("info", "- Mixing & homogeneity: standardize vortex speed/time, centrifuge/settle times, and ensure suspensions are uniformly mixed before aliquoting."))
            impacts.append(("info", "- Environmental control: minimize humidity/temperature swings and evaporation—microvolume assays are especially sensitive."))
            impacts.append(("info", "- Lock analysis settings: method version, peak integration/thresholds, curve-fit settings; avoid manual, subjective adjustments."))

        else:
            impacts.append((
                "info",
                f"- Reproducibility is the dominant source ({pct_repro:.1f}% vs {pct_repeat:.1f}%). "
                "Suggests issues with operator differences (training, technique)."
            ))

            # Reproducibility troubleshooting tips (between-operator variation)
            impacts.append(("info", "Reproducibility troubleshooting (between-operator variation):"))
            impacts.append(("info", "- Standardize technique: pipetting angle/speed, mixing style, timing, and plate handling; do a brief side-by-side demo using the same sample."))
            impacts.append(("info", "- Tighten work instructions: specify critical steps and tolerances (e.g., vortex 10 ±2 s; incubate 5:00 ±0:30; define 'room temperature')."))
            impacts.append(("info", "- Standardize dilution paths: mandate the exact dilution scheme and order-of-addition; consider reverse pipetting for viscous liquids."))
            impacts.append(("info", "- Training with objective checks: use QC targets/controls to qualify operators before production runs."))
            impacts.append(("info", "- Reduce interpretation: automate dilution/math and lock analysis settings to remove subjective choices."))
            impacts.append(("info", "- Balance/randomize assignments: ensure each operator measures a balanced mix of parts to avoid operator/part confounding."))

    return impacts


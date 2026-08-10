from datetime import datetime


def generate_markdown_report(
    title,
    summary,
    analysis,
    entities,
    entity_metrics,
    research_metrics,
):
    """
    Generate a downloadable Markdown research report.

    Args:
        title: Research paper title.
        summary: Transformer-generated summary.
        analysis: Structured ResearchAnalysis object.
        entities: Biomedical entity DataFrame.
        entity_metrics: Entity-level metrics dictionary.
        research_metrics: Research-analysis metrics dictionary.

    Returns:
        Markdown report as a string.
    """

    generated_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    lines = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"**Generated:** {generated_at}"
    )
    lines.append("")

    # --------------------------------------------------
    # Executive Summary
    # --------------------------------------------------

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary or "Not available.")
    lines.append("")

    # --------------------------------------------------
    # Research Objective
    # --------------------------------------------------

    lines.append("## Research Objective")
    lines.append("")
    lines.append(
        analysis.research_objective
    )
    lines.append("")

    # --------------------------------------------------
    # Methodology
    # --------------------------------------------------

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        analysis.methodology
    )
    lines.append("")

    # --------------------------------------------------
    # Key Findings
    # --------------------------------------------------

    lines.append("## Key Findings")
    lines.append("")

    for finding in analysis.key_findings:
        lines.append(f"- {finding}")

    lines.append("")

    # --------------------------------------------------
    # Limitations
    # --------------------------------------------------

    lines.append("## Limitations")
    lines.append("")

    for limitation in analysis.limitations:
        lines.append(f"- {limitation}")

    lines.append("")

    # --------------------------------------------------
    # Research Gaps
    # --------------------------------------------------

    lines.append("## Research Gaps")
    lines.append("")

    for gap in analysis.research_gaps:
        lines.append(f"- {gap}")

    lines.append("")

    # --------------------------------------------------
    # Future Directions
    # --------------------------------------------------

    lines.append("## Future Directions")
    lines.append("")

    for direction in analysis.future_directions:
        lines.append(f"- {direction}")

    lines.append("")

    # --------------------------------------------------
    # Biomedical Entity Analysis
    # --------------------------------------------------

    lines.append("## Biomedical Entity Analysis")
    lines.append("")

    lines.append(
        f"- Total entities: "
        f"{entity_metrics['total_entities']}"
    )

    lines.append(
        f"- Unique entities: "
        f"{entity_metrics['unique_entities']}"
    )

    lines.append(
        f"- Diseases detected: "
        f"{entity_metrics['disease_count']}"
    )

    lines.append(
        f"- Chemicals detected: "
        f"{entity_metrics['chemical_count']}"
    )

    lines.append(
        f"- Average confidence: "
        f"{entity_metrics['average_confidence']}"
    )

    lines.append("")

    # --------------------------------------------------
    # Research Analysis Metrics
    # --------------------------------------------------

    lines.append("## Research Analysis Metrics")
    lines.append("")

    lines.append(
        f"- Key findings: "
        f"{research_metrics['findings_count']}"
    )

    lines.append(
        f"- Limitations: "
        f"{research_metrics['limitations_count']}"
    )

    lines.append(
        f"- Research gaps: "
        f"{research_metrics['research_gaps_count']}"
    )

    lines.append(
        f"- Future directions: "
        f"{research_metrics['future_directions_count']}"
    )

    lines.append("")

    # --------------------------------------------------
    # Entity Table
    # --------------------------------------------------

    lines.append("## Extracted Biomedical Entities")
    lines.append("")

    if entities is not None and not entities.empty:

        lines.append(
            "| Entity | Category | Confidence |"
        )
        lines.append(
            "|---|---|---:|"
        )

        for _, row in entities.iterrows():

            lines.append(
                f"| {row['Entity']} "
                f"| {row['Category']} "
                f"| {row['Confidence']:.3f} |"
            )

    else:

        lines.append(
            "No biomedical entities were detected."
        )

    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Generated by Biomedical Research Intelligence Platform."
    )

    return "\n".join(lines)

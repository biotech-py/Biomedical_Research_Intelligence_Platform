import pandas as pd


def calculate_entity_metrics(entities):
    """
    Calculate quantitative metrics from biomedical entities.

    Args:
        entities: DataFrame containing Entity, Category, Confidence.

    Returns:
        Dictionary containing entity-level metrics.
    """

    if entities is None or entities.empty:
        return {
            "total_entities": 0,
            "unique_entities": 0,
            "disease_count": 0,
            "chemical_count": 0,
            "average_confidence": 0.0,
        }

    total_entities = len(entities)

    unique_entities = (
        entities["Entity"]
        .str.lower()
        .nunique()
    )

    disease_count = int(
        (entities["Category"] == "Disease").sum()
    )

    chemical_count = int(
        (entities["Category"] == "Chemical").sum()
    )

    average_confidence = round(
        entities["Confidence"].mean(),
        3
    )

    return {
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "disease_count": disease_count,
        "chemical_count": chemical_count,
        "average_confidence": average_confidence,
    }


def calculate_research_metrics(analysis):
    """
    Calculate quantitative metrics from the structured
    research analysis.

    Args:
        analysis: ResearchAnalysis object.

    Returns:
        Dictionary containing research-analysis metrics.
    """

    return {
        "findings_count": len(
            analysis.key_findings
        ),
        "limitations_count": len(
            analysis.limitations
        ),
        "research_gaps_count": len(
            analysis.research_gaps
        ),
        "future_directions_count": len(
            analysis.future_directions
        ),
    }


def build_category_distribution(entities):
    """
    Create a category distribution DataFrame
    for visualization.
    """

    if entities is None or entities.empty:
        return pd.DataFrame(
            columns=["Category", "Count"]
        )

    distribution = (
        entities["Category"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Category",
        "Count",
    ]

    return distribution


def build_entity_frequency(entities):
    """
    Create an entity-frequency DataFrame
    for visualization.
    """

    if entities is None or entities.empty:
        return pd.DataFrame(
            columns=["Entity", "Frequency"]
        )

    frequency = (
        entities["Entity"]
        .value_counts()
        .reset_index()
    )

    frequency.columns = [
        "Entity",
        "Frequency",
    ]

    return frequency

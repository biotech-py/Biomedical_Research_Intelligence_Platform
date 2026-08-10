import pandas as pd


def extract_entities(
    text,
    biomedical_ner,
    confidence_threshold=0.70
):
    """
    Extract biomedical disease and chemical entities.

    Args:
        text: Biomedical research text.
        biomedical_ner: Loaded biomedical NER pipeline.
        confidence_threshold: Minimum confidence required.

    Returns:
        DataFrame containing:
        Entity, Category, Confidence
    """

    columns = [
        "Entity",
        "Category",
        "Confidence"
    ]

    text = text.strip()

    if not text:
        return pd.DataFrame(columns=columns)

    results = biomedical_ner(text)

    entity_data = []

    for entity in results:

        word = entity.get("word", "").strip()
        category = entity.get("entity_group", "")
        score = float(entity.get("score", 0))

        # Ignore low-confidence predictions.
        if score < confidence_threshold:
            continue

        # Ignore tokenizer subword fragments.
        if word.startswith("##"):
            continue

        # Normalize whitespace introduced by tokenization.
        word = " ".join(word.split())

        if not word:
            continue

        entity_data.append(
            {
                "Entity": word,
                "Category": category,
                "Confidence": round(score, 3)
            }
        )

    entities = pd.DataFrame(
        entity_data,
        columns=columns
    )

    if entities.empty:
        return entities

    # Remove exact duplicate predictions.
    entities = entities.drop_duplicates(
        subset=["Entity", "Category"]
    ).reset_index(drop=True)

    return entities


def get_entity_counts(entities):
    """
    Calculate the frequency of each biomedical entity.
    """

    if entities.empty:
        return pd.DataFrame(
            columns=["Entity", "Frequency"]
        )

    counts = (
        entities["Entity"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Entity",
        "Frequency"
    ]

    return counts


def get_category_counts(entities):
    """
    Calculate the frequency of each biomedical entity category.
    """

    if entities.empty:
        return pd.DataFrame(
            columns=["Category", "Count"]
        )

    counts = (
        entities["Category"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Category",
        "Count"
    ]

    return counts

from transformers import pipeline


SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
BIOMEDICAL_NER_MODEL = "Glasgow-AI4BioMed/bioner_bc5cdr"


def load_models():
    """
    Load the Transformer models required by the application.

    Returns:
        tuple:
            summarizer: Transformer summarization pipeline
            biomedical_ner: Biomedical NER pipeline
    """

    summarizer = pipeline(
        "summarization",
        model=SUMMARIZATION_MODEL
    )

    biomedical_ner = pipeline(
        "token-classification",
        model=BIOMEDICAL_NER_MODEL,
        aggregation_strategy="simple"
    )

    return summarizer, biomedical_ner

import os

from google import genai
from pydantic import BaseModel, Field


GEMINI_MODEL = "gemini-3.5-flash"


class ResearchAnalysis(BaseModel):
    research_objective: str = Field(
        description=(
            "The main research problem, question, or objective "
            "addressed by the study."
        )
    )

    methodology: str = Field(
        description=(
            "A concise description of the experimental, computational, "
            "clinical, or analytical methods used in the study."
        )
    )

    key_findings: list[str] = Field(
        description=(
            "Important findings or results explicitly supported "
            "by the supplied research text."
        )
    )

    limitations: list[str] = Field(
        description=(
            "Limitations explicitly reported by the authors or clearly "
            "supported by the supplied text. Do not invent limitations."
        )
    )

    research_gaps: list[str] = Field(
        description=(
            "Research questions, unresolved issues, or areas requiring "
            "further investigation based on the supplied text."
        )
    )

    future_directions: list[str] = Field(
        description=(
            "Potential future research directions grounded in the study's "
            "findings, limitations, or unresolved questions."
        )
    )


def get_client():
    """
    Create and return a Gemini API client.

    The API key is read from the GEMINI_API_KEY
    environment variable.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(api_key=api_key)


def analyze_research(
    text,
    summary="",
    entities=None
):
    """
    Analyze biomedical research text using Gemini.

    Args:
        text: Original research text.
        summary: Transformer-generated summary.
        entities: Extracted biomedical entities.

    Returns:
        ResearchAnalysis object.
    """

    text = text.strip()

    if not text:
        raise ValueError(
            "Research text cannot be empty."
        )

    client = get_client()

    if entities is not None and not entities.empty:

        entity_records = entities[
            ["Entity", "Category", "Confidence"]
        ].to_dict("records")

    else:
        entity_records = []

    prompt = f"""
You are a biomedical research analysis assistant.

Analyze ONLY the information provided below.

Your task is to identify the research objective,
methodology, key findings, limitations, research gaps,
and future research directions.

IMPORTANT RULES:

1. Do not invent experimental results.
2. Do not invent sample sizes, statistics, methods,
   citations, or conclusions.
3. Distinguish clearly between reported findings and
   reasonable future research directions.
4. If a limitation is not explicitly reported or
   reasonably supported by the supplied text, say:
   "Not clearly reported in the supplied text."
5. Research gaps should be grounded in unresolved
   questions or limitations present in the text.
6. Future directions should be plausible extensions
   of the supplied research, not unrelated ideas.
7. Keep the analysis concise but informative.
8. Use scientific terminology where appropriate.

TRANSFORMER SUMMARY:
{summary}

BIOMEDICAL ENTITIES:
{entity_records}

RESEARCH TEXT:
{text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ResearchAnalysis,
        },
    )

    return ResearchAnalysis.model_validate_json(
        response.text
    )

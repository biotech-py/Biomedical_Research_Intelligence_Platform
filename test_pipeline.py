from src.models import load_models
from src.summarizer import summarize_text
from src.biomedical_ner import extract_entities
from src.research_analyzer import analyze_research
from src.analytics import (
    calculate_entity_metrics,
    calculate_research_metrics,
)
from src.report_generator import generate_markdown_report


SAMPLE_RESEARCH = """
Magnesium alloys are being investigated as biodegradable materials
for temporary orthopedic implants because they can gradually degrade
inside the human body. However, the rapid corrosion of magnesium can
lead to excessive hydrogen evolution, increased local pH, and premature
loss of mechanical integrity.

This study investigated the use of polymer coatings to improve the
corrosion resistance of AZ31B magnesium alloy. AZ31B samples were
surface treated and coated with a biodegradable polymer before being
immersed in a simulated physiological solution. Corrosion behavior
was evaluated using hydrogen evolution measurements and
electrochemical characterization.

The coated samples demonstrated lower hydrogen evolution and improved
corrosion resistance compared with untreated AZ31B magnesium.
Electrochemical measurements also indicated improved surface
protection after coating.

The findings suggest that polymer coatings may help control the
corrosion rate of biodegradable magnesium implants and potentially
extend their functional lifetime.
"""


def main():

    print("=" * 70)
    print("BIOMEDICAL RESEARCH INTELLIGENCE PLATFORM")
    print("END-TO-END BACKEND TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load AI models
    # --------------------------------------------------

    print("\n[1/5] Loading AI models...")

    summarizer, biomedical_ner = load_models()

    print("Models loaded successfully.")

    # --------------------------------------------------
    # 2. Generate summary
    # --------------------------------------------------

    print("\n[2/5] Generating research summary...")

    summary = summarize_text(
        SAMPLE_RESEARCH,
        summarizer
    )

    print("\nSUMMARY:")
    print("-" * 70)
    print(summary)

    # --------------------------------------------------
    # 3. Extract biomedical entities
    # --------------------------------------------------

    print("\n[3/5] Extracting biomedical entities...")

    entities = extract_entities(
        SAMPLE_RESEARCH,
        biomedical_ner
    )

    print("\nBIOMEDICAL ENTITIES:")
    print("-" * 70)

    if entities.empty:
        print("No biomedical entities detected.")
    else:
        print(
            entities.to_string(index=False)
        )

    # --------------------------------------------------
    # 4. Gemini research analysis
    # --------------------------------------------------

    print("\n[4/5] Running Gemini research analysis...")

    analysis = analyze_research(
        text=SAMPLE_RESEARCH,
        summary=summary,
        entities=entities,
    )

    print("\nRESEARCH ANALYSIS:")
    print("-" * 70)

    print("\nResearch Objective:")
    print(analysis.research_objective)

    print("\nMethodology:")
    print(analysis.methodology)

    print("\nKey Findings:")

    for finding in analysis.key_findings:
        print(f"- {finding}")

    print("\nLimitations:")

    for limitation in analysis.limitations:
        print(f"- {limitation}")

    print("\nResearch Gaps:")

    for gap in analysis.research_gaps:
        print(f"- {gap}")

    print("\nFuture Directions:")

    for direction in analysis.future_directions:
        print(f"- {direction}")

    # --------------------------------------------------
    # 5. Analytics
    # --------------------------------------------------

    entity_metrics = calculate_entity_metrics(
        entities
    )

    research_metrics = calculate_research_metrics(
        analysis
    )

    print("\nANALYTICS:")
    print("-" * 70)

    print("\nEntity Metrics:")
    print(entity_metrics)

    print("\nResearch Metrics:")
    print(research_metrics)

    # --------------------------------------------------
    # 6. Generate report
    # --------------------------------------------------

    print("\nGenerating research report...")

    report = generate_markdown_report(
        title="AZ31B Magnesium Alloy Polymer Coating Study",
        summary=summary,
        analysis=analysis,
        entities=entities,
        entity_metrics=entity_metrics,
        research_metrics=research_metrics,
    )

    output_path = "outputs/test_research_report.md"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print(
        f"\nReport generated successfully:"
    )

    print(output_path)

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
   
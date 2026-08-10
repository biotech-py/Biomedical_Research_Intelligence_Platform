import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models import load_models
from src.summarizer import summarize_text
from src.biomedical_ner import extract_entities
from src.research_analyzer import analyze_research
from src.analytics import (
    calculate_entity_metrics,
    calculate_research_metrics,
)
from src.report_generator import generate_markdown_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Biomedical Research Intelligence Platform",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🧬 Biomedical Research Intelligence Platform")

st.markdown(
    """
Analyze biomedical literature using **Transformer-based NLP,
biomedical named entity recognition, and Generative AI**.

The platform generates:
- Executive summaries
- Biomedical entities
- Research objectives and methodology
- Key findings
- Limitations and research gaps
- Future research directions
- Literature analytics
- Downloadable research reports
"""
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def initialize_models():
    return load_models()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    st.markdown(
        """
### AI Pipeline

**1. Transformer**
        
Research summarization

**2. Biomedical NER**

Disease and chemical extraction

**3. Gemini**

Research intelligence analysis

**4. Analytics**

Quantitative literature analysis
"""
    )

    st.divider()

    gemini_available = bool(
        os.getenv("GEMINI_API_KEY")
    )

    if gemini_available:
        st.success(
            "Gemini API connected"
        )
    else:
        st.warning(
            "Gemini API key not detected"
        )


# ============================================================
# INPUT
# ============================================================

st.header("📄 Research Input")

st.markdown(
    "Paste a biomedical research abstract, paper section, "
    "or research text below."
)

research_text = st.text_area(
    "Research Text",
    height=300,
    placeholder=(
        "Paste biomedical research text here..."
    ),
)


analyze_button = st.button(
    "🔬 Analyze Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

if analyze_button:

    if not research_text.strip():

        st.error(
            "Please enter biomedical research text before analysis."
        )

        st.stop()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    with st.spinner(
        "Loading biomedical NLP models..."
    ):

        summarizer_model, ner_model = (
            initialize_models()
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with st.spinner(
        "Generating research summary..."
    ):

        try:

            summary = summarize_text(
                research_text,
                summarizer_model,
            )

        except Exception as error:

            st.error(
                f"Summary generation failed: {error}"
            )

            st.stop()

    # --------------------------------------------------------
    # Biomedical NER
    # --------------------------------------------------------

    with st.spinner(
        "Extracting biomedical entities..."
    ):

        try:

            entities = extract_entities(
                research_text,
                ner_model,
            )

        except Exception as error:

            st.error(
                f"Biomedical entity extraction failed: {error}"
            )

            st.stop()

    # --------------------------------------------------------
    # Gemini Research Analysis
    # --------------------------------------------------------

    analysis = None

    if gemini_available:

        with st.spinner(
            "Generating research intelligence with Gemini..."
        ):

            try:

                analysis = analyze_research(
                    text=research_text,
                    summary=summary,
                    entities=entities,
                )

            except Exception as error:

                st.warning(
                    "Gemini research analysis is currently "
                    "unavailable."
                )

                st.caption(
                    f"Technical detail: {error}"
                )

    else:

        st.info(
            "Gemini research analysis is unavailable because "
            "GEMINI_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    entity_metrics = (
        calculate_entity_metrics(
            entities
        )
    )

    research_metrics = None

    if analysis is not None:

        research_metrics = (
            calculate_research_metrics(
                analysis
            )
        )

    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Research Analysis")

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    st.subheader("📝 Executive Summary")

    st.write(summary)

    # --------------------------------------------------------
    # Biomedical Entities
    # --------------------------------------------------------

    st.subheader(
        "🧬 Biomedical Entities"
    )

    if entities.empty:

        st.info(
            "No biomedical entities were detected."
        )

    else:

        display_entities = entities.copy()

        display_entities[
            "Confidence"
        ] = display_entities[
            "Confidence"
        ].round(3)

        st.dataframe(
            display_entities,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Entity Metrics
    # --------------------------------------------------------

    st.subheader(
        "📈 Entity Analytics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Entities",
            entity_metrics[
                "total_entities"
            ],
        )

    with col2:

        st.metric(
            "Unique Entities",
            entity_metrics[
                "unique_entities"
            ],
        )

    with col3:

        st.metric(
            "Diseases",
            entity_metrics[
                "disease_count"
            ],
        )

    with col4:

        st.metric(
            "Chemicals",
            entity_metrics[
                "chemical_count"
            ],
        )

    # --------------------------------------------------------
    # Entity Distribution
    # --------------------------------------------------------

    if not entities.empty:

        category_counts = (
            entities[
                "Category"
            ]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Count",
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title="Biomedical Entity Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # GEMINI RESEARCH INTELLIGENCE
    # ========================================================

    if analysis is not None:

        st.divider()

        st.header(
            "🧠 Generative AI Research Intelligence"
        )

        # ----------------------------------------------------
        # Objective
        # ----------------------------------------------------

        st.subheader(
            "🎯 Research Objective"
        )

        st.write(
            analysis.research_objective
        )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        st.subheader(
            "🔬 Methodology"
        )

        st.write(
            analysis.methodology
        )

        # ----------------------------------------------------
        # Key Findings
        # ----------------------------------------------------

        st.subheader(
            "📌 Key Findings"
        )

        for finding in analysis.key_findings:

            st.markdown(
                f"- {finding}"
            )

        # ----------------------------------------------------
        # Limitations
        # ----------------------------------------------------

        st.subheader(
            "⚠️ Limitations"
        )

        for limitation in analysis.limitations:

            st.markdown(
                f"- {limitation}"
            )

        # ----------------------------------------------------
        # Research Gaps
        # ----------------------------------------------------

        st.subheader(
            "🔎 Research Gaps"
        )

        for gap in analysis.research_gaps:

            st.markdown(
                f"- {gap}"
            )

        # ----------------------------------------------------
        # Future Directions
        # ----------------------------------------------------

        st.subheader(
            "🚀 Future Directions"
        )

        for direction in analysis.future_directions:

            st.markdown(
                f"- {direction}"
            )

        # ----------------------------------------------------
        # Research Metrics
        # ----------------------------------------------------

        if research_metrics:

            st.subheader(
                "📊 Research Analysis Metrics"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Key Findings",
                    research_metrics[
                        "findings_count"
                    ],
                )

            with col2:

                st.metric(
                    "Limitations",
                    research_metrics[
                        "limitations_count"
                    ],
                )

            with col3:

                st.metric(
                    "Research Gaps",
                    research_metrics[
                        "research_gaps_count"
                    ],
                )

            with col4:

                st.metric(
                    "Future Directions",
                    research_metrics[
                        "future_directions_count"
                    ],
                )

        # ====================================================
        # REPORT
        # ====================================================

        st.divider()

        st.header(
            "📥 Download Research Report"
        )

        report = generate_markdown_report(
            title="Biomedical Research Analysis",
            summary=summary,
            analysis=analysis,
            entities=entities,
            entity_metrics=entity_metrics,
            research_metrics=research_metrics,
        )

        st.download_button(
            label="⬇️ Download Research Report",
            data=report,
            file_name="biomedical_research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    else:

        st.info(
            "Generative AI research analysis is unavailable. "
            "Summary, biomedical NER, and analytics are still "
            "available."
        )
        
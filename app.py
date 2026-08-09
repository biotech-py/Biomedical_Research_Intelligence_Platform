import streamlit as st
import pandas as pd
import plotly.express as px
import re
from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Biomedical Research Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧬 Research Intelligence")

    st.markdown("---")

    st.markdown(
        """
        ### Features

        ✅ Transformer-Based Summarization

        ✅ Biomedical Named Entity Recognition

        ✅ Biomedical Entity Classification

        ✅ Research Insight Generation

        ✅ Literature Analytics

        ✅ Interactive Visualization

        ✅ Downloadable Analysis Report
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### NLP Workflow

        📄 Biomedical Literature

        ↓

        🤖 Transformer Summarization

        ↓

        🧬 Biomedical NER

        ↓

        📊 Literature Analytics

        ↓

        💡 Research Insights
        """
    )

    st.markdown("---")

    if st.button("📖 Load Sample Abstract"):

        st.session_state.sample_text = """
        The EGFR T790M mutation is one of the major causes of acquired
        resistance to first-generation EGFR tyrosine kinase inhibitors
        in non-small cell lung cancer. Osimertinib has emerged as an
        effective therapeutic strategy targeting EGFR T790M-positive
        tumors. Clinical studies have demonstrated improved response
        rates in patients with EGFR T790M-positive disease.
        """


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    # Transformer-based text summarization model
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    # Biomedical Named Entity Recognition model
    biomedical_ner = pipeline(
        "token-classification",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple"
    )

    return summarizer, biomedical_ner


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading NLP models..."):

    try:

        summarizer, biomedical_ner = load_models()

        models_loaded = True

    except Exception as e:

        models_loaded = False

        summarizer = None
        biomedical_ner = None

        st.error(
            "Unable to load the NLP models. "
            "Please check the model dependencies and internet connection."
        )


# ============================================================
# HERO SECTION
# ============================================================

st.title("🧬 Biomedical Research Intelligence Platform")

st.markdown(
    """
    ### AI-Powered Biomedical Literature Analysis

    An NLP-based platform for automated biomedical literature
    summarization, biomedical entity extraction, and research analytics.
    """
)

st.markdown("---")


# ============================================================
# INPUT SECTION
# ============================================================

default_text = st.session_state.get("sample_text", "")

text = st.text_area(
    "📄 Paste Biomedical Abstract or Research Text",
    value=default_text,
    height=250,
    placeholder="Paste a biomedical research abstract here..."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_biomedical_text(text):

    biomedical_terms = [
        "gene",
        "protein",
        "cell",
        "cancer",
        "tumor",
        "dna",
        "rna",
        "drug",
        "therapy",
        "patient",
        "mutation",
        "disease",
        "treatment",
        "biomarker",
        "egfr",
        "kras",
        "tp53",
        "clinical",
        "molecular",
        "pathway",
        "receptor",
        "inhibitor",
        "diagnosis",
        "oncology",
        "medicine"
    ]

    text_lower = text.lower()

    matches = [
        term for term in biomedical_terms
        if re.search(rf"\b{re.escape(term)}\b", text_lower)
    ]

    return len(matches) > 0


def generate_summary(text):

    # Transformer models have practical input limits.
    # We therefore use the first portion of the text for summarization.
    words = text.split()

    if len(words) < 50:
        return text.strip()

    limited_text = " ".join(words[:900])

    result = summarizer(
        limited_text,
        max_length=150,
        min_length=40,
        do_sample=False
    )

    return result[0]["summary_text"]


def extract_entities(text):

    results = biomedical_ner(text)

    entity_data = []

    for entity in results:

        entity_data.append(
            {
                "Entity": entity["word"],
                "Category": entity["entity_group"],
                "Confidence": round(float(entity["score"]), 3)
            }
        )

    return pd.DataFrame(entity_data)


def generate_research_insight(df_entities, text):

    if df_entities.empty:

        return (
            "No biomedical entities were confidently detected "
            "in the provided text."
        )

    entity_categories = (
        df_entities["Category"]
        .value_counts()
        .to_dict()
    )

    top_entities = (
        df_entities["Entity"]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    category_text = ", ".join(
        [
            f"{category} ({count})"
            for category, count in entity_categories.items()
        ]
    )

    entity_text = ", ".join(top_entities)

    insight = f"""
    The literature contains {len(df_entities)} detected biomedical
    entity mentions across the following categories: {category_text}.

    The most frequently detected entities include:
    {entity_text}.

    These entities provide an overview of the major biological,
    molecular, therapeutic, or clinical concepts represented in
    the analyzed literature.
    """

    return insight.strip()


def create_report(
    text,
    summary,
    df_entities,
    insight
):

    report = []

    report.append(
        "BIOMEDICAL RESEARCH INTELLIGENCE REPORT"
    )

    report.append("=" * 60)

    report.append("\nORIGINAL TEXT")
    report.append("-" * 60)
    report.append(text)

    report.append("\n\nTRANSFORMER-GENERATED SUMMARY")
    report.append("-" * 60)
    report.append(summary)

    report.append("\n\nBIOMEDICAL ENTITIES")
    report.append("-" * 60)

    if not df_entities.empty:

        report.append(
            df_entities.to_string(index=False)
        )

    else:

        report.append(
            "No biomedical entities detected."
        )

    report.append("\n\nRESEARCH INSIGHT")
    report.append("-" * 60)
    report.append(insight)

    report.append("\n\nLITERATURE ANALYTICS")
    report.append("-" * 60)

    report.append(
        f"Total Words: {len(text.split())}"
    )

    report.append(
        f"Total Entities: {len(df_entities)}"
    )

    if not df_entities.empty:

        avg_confidence = df_entities["Confidence"].mean()

        report.append(
            f"Average Entity Confidence: "
            f"{avg_confidence:.3f}"
        )

    return "\n".join(report)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Literature", type="primary"):

    if not text.strip():

        st.warning(
            "Please enter a biomedical abstract or research text."
        )

        st.stop()


    # --------------------------------------------------------
    # Biomedical input validation
    # --------------------------------------------------------

    if not is_biomedical_text(text):

        st.error(
            "The provided text does not appear to contain "
            "recognizable biomedical content."
        )

        st.stop()


    if not models_loaded:

        st.error(
            "NLP models could not be loaded. "
            "Please restart the application and try again."
        )

        st.stop()


    # --------------------------------------------------------
    # Transformer summarization
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Generating Transformer-based summary..."
    ):

        try:

            summary = generate_summary(text)

        except Exception as e:

            st.error(
                f"Summarization failed: {str(e)}"
            )

            summary = text


    # --------------------------------------------------------
    # Biomedical NER
    # --------------------------------------------------------

    with st.spinner(
        "🧬 Detecting biomedical entities..."
    ):

        try:

            df_entities = extract_entities(text)

        except Exception as e:

            st.error(
                f"Biomedical entity extraction failed: {str(e)}"
            )

            df_entities = pd.DataFrame(
                columns=[
                    "Entity",
                    "Category",
                    "Confidence"
                ]
            )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_entities = len(df_entities)

    total_words = len(text.split())

    total_sentences = len(
        [
            sentence
            for sentence in re.split(r"[.!?]+", text)
            if sentence.strip()
        ]
    )

    if not df_entities.empty:

        avg_confidence = df_entities[
            "Confidence"
        ].mean()

    else:

        avg_confidence = 0


    # --------------------------------------------------------
    # Research Insight
    # --------------------------------------------------------

    insight = generate_research_insight(
        df_entities,
        text
    )


    # ========================================================
    # METRICS
    # ========================================================

    st.subheader(
        "📊 Research Intelligence Metrics"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "🧬 Entities",
            total_entities
        )

    with c2:

        st.metric(
            "📄 Words",
            total_words
        )

    with c3:

        st.metric(
            "📝 Sentences",
            total_sentences
        )

    with c4:

        st.metric(
            "🎯 Avg Confidence",
            f"{avg_confidence:.2f}"
        )


    st.markdown("---")


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🤖 AI Summary",
            "🧬 Biomedical Entities",
            "📊 Literature Analytics",
            "💡 Research Insights"
        ]
    )


    # ========================================================
    # TAB 1 — SUMMARY
    # ========================================================

    with tab1:

        st.subheader(
            "📄 Original Biomedical Text"
        )

        st.write(text)


        st.markdown("---")


        st.subheader(
            "🤖 Transformer-Generated Summary"
        )

        st.success(summary)


        st.markdown("---")


        report = create_report(
            text,
            summary,
            df_entities,
            insight
        )


        st.download_button(
            label="📥 Download Full Analysis Report",
            data=report,
            file_name="biomedical_research_analysis.txt",
            mime="text/plain"
        )


    # ========================================================
    # TAB 2 — ENTITIES
    # ========================================================

    with tab2:

        st.subheader(
            "🧬 Biomedical Named Entity Recognition"
        )

        st.markdown(
            """
            Biomedical entities are automatically extracted
            using a Transformer-based biomedical NER model.
            """
        )


        if not df_entities.empty:

            st.dataframe(
                df_entities,
                use_container_width=True,
                hide_index=True
            )


            st.markdown("---")

            st.subheader(
                "Top Detected Entities"
            )

            entity_frequency = (
                df_entities["Entity"]
                .value_counts()
                .head(10)
                .reset_index()
            )

            entity_frequency.columns = [
                "Entity",
                "Frequency"
            ]

            st.dataframe(
                entity_frequency,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No biomedical entities were detected."
            )


    # ========================================================
    # TAB 3 — ANALYTICS
    # ========================================================

    with tab3:

        st.subheader(
            "📈 Literature Analytics"
        )


        if not df_entities.empty:

            # ------------------------------------------------
            # Entity category distribution
            # ------------------------------------------------

            category_counts = (
                df_entities["Category"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "Category",
                "Count"
            ]


            fig_category = px.bar(
                category_counts,
                x="Category",
                y="Count",
                title="Biomedical Entity Distribution",
                labels={
                    "Count": "Entity Count",
                    "Category": "Entity Category"
                }
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True
            )


            # ------------------------------------------------
            # Entity frequency
            # ------------------------------------------------

            entity_counts = (
                df_entities["Entity"]
                .value_counts()
                .head(10)
                .reset_index()
            )

            entity_counts.columns = [
                "Entity",
                "Frequency"
            ]


            fig_entities = px.bar(
                entity_counts,
                x="Entity",
                y="Frequency",
                title="Top Biomedical Entities",
                labels={
                    "Frequency": "Frequency",
                    "Entity": "Biomedical Entity"
                }
            )


            st.plotly_chart(
                fig_entities,
                use_container_width=True
            )


            # ------------------------------------------------
            # Summary metrics
            # ------------------------------------------------

            st.markdown("---")

            a1, a2, a3 = st.columns(3)

            with a1:

                st.metric(
                    "Unique Entities",
                    df_entities["Entity"]
                    .str.lower()
                    .nunique()
                )

            with a2:

                st.metric(
                    "Entity Categories",
                    df_entities["Category"]
                    .nunique()
                )

            with a3:

                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.3f}"
                )


        else:

            st.warning(
                "No entities available for analytics."
            )


    # ========================================================
    # TAB 4 — RESEARCH INSIGHTS
    # ========================================================

    with tab4:

        st.subheader(
            "💡 Research Insights"
        )

        st.info(insight)


        if not df_entities.empty:

            st.markdown("---")

            st.subheader(
                "🔬 Key Biomedical Concepts"
            )

            top_entities = (
                df_entities["Entity"]
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

            for entity in top_entities:

                st.markdown(
                    f"- **{entity}**"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Developed by Nirupam Joarder | "
    "Biomedical Research Intelligence Platform | "
    "Transformer-Based Biomedical NLP"
)

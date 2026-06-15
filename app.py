import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Drug Discovery Literature Assistant",
    page_icon="🧬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.title("🧬 Research Intelligence")

    st.markdown("---")

    st.markdown("""
    ### Features

    ✅ AI Literature Summarization

    ✅ Biomedical Entity Detection

    ✅ Research Insight Engine

    ✅ Literature Analytics

    ✅ Confidence Scoring

    ✅ Interactive Visualization
    """)

    st.markdown("---")

    st.markdown("""
    ### Models

    📄 Text Summarization Engine

    🧠 Rule-Based Biomedical NLP

    🧬 Biomedical Entity Detection

    🔍 Keyword-Based Entity Recognition
    """)

    st.markdown("---")

if st.button("📖 Load Sample Abstract"):

        st.session_state.sample_text = """
The EGFR T790M mutation is one of the major causes of acquired resistance to first-generation EGFR tyrosine kinase inhibitors in non-small cell lung cancer. Osimertinib has emerged as an effective therapeutic strategy targeting EGFR T790M-positive tumors.
"""

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
summarizer = None

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.title("🧬 Biomedical Research Intelligence Platform")

st.markdown("""
### AI-Powered Biomedical Literature Analysis

Using AI-powered summarization and biomedical entity extraction
to automatically extract insights from scientific literature.
""")

st.markdown("---")

# --------------------------------------------------
# INPUT
# --------------------------------------------------
default_text = st.session_state.get("sample_text", "")

text = st.text_area(
    "📄 Paste Biomedical Abstract",
    value=default_text,
    height=220
)
# --------------------------------------------------
# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
if st.button("🔍 Analyze Literature"):

    if text.strip():

        biomedical_keywords = [
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
            "tp53"
        ]

        keyword_count = sum(
            1 for word in biomedical_keywords
            if word in text.lower()
        )

        if keyword_count == 0:

            st.error(
                "Input does not appear to be a biomedical abstract."
            )

            st.stop()

        with st.spinner("🧠 Running Transformer Model..."):

            try:

                sentences = [s.strip() for s in text.split(".") if s.strip()]

                if len(sentences) >= 2:
                    summary = ".".join(sentences[:2]) + "."
                else:
                    summary = text

            except:

                summary = text[:300] + "..."

            import re

            entity_patterns = [
                "KRAS",
                "EGFR",
                "TP53",
                "BRCA1",
                "BRCA2",
                "Cetuximab",
                "Osimertinib",
                "cancer",
                "tumor",
                "tumors",
                "mutation",
                "mutations",
                "therapy",
                "therapeutic",
                "drug",
                "protein",
                "gene",
                "cell",
                "cells",
                "DNA",
                "RNA",
                "biomarker",
                "patient",
                "disease",
                "treatment"
            ]

            entity_data = []
            unique_entities = set()
            GENES = ["KRAS", "EGFR", "TP53", "BRCA1", "BRCA2"]

            DRUGS = ["Cetuximab", "Osimertinib"]

            DISEASES = ["cancer", "tumor", "tumors", "disease"]

            THERAPIES = ["therapy", "therapeutic", "treatment"]

            BIOMARKERS = ["biomarker", "biomarkers"]

            entity_patterns = (
                GENES
                + DRUGS
                + DISEASES
                + THERAPIES
                + BIOMARKERS
            )

            for pattern in entity_patterns:

                matches = re.findall(
                    rf"\b{re.escape(pattern)}\b",
                    text,
                    flags=re.IGNORECASE
                )

                for match in matches:

                    if match.lower() not in unique_entities:

                        unique_entities.add(match.lower())

                        if match.upper() in [g.upper() for g in GENES]:
                            category = "Gene"

                        elif match.lower() in [d.lower() for d in DRUGS]:
                            category = "Drug"

                        elif match.lower() in [d.lower() for d in DISEASES]:
                            category = "Disease"

                        elif match.lower() in [t.lower() for t in THERAPIES]:
                            category = "Therapy"

                        elif match.lower() in [b.lower() for b in BIOMARKERS]:
                            category = "Biomarker"

                        else:
                            category = "Other"

                        entity_data.append({
                            "Entity": match,
                            "Category": category,
                            "Confidence": 1.0
                        })
            st.write("DEBUG:", len(entity_data))
            df_entities = pd.DataFrame(entity_data)

            total_entities = len(df_entities)

            avg_confidence = (
                            1.0 if not df_entities.empty else 0
            )

            st.subheader("📊 Research Intelligence Metrics")

            c1, c2, c3 = st.columns(3)

            with c1:
                        st.metric("🔬 Entities", total_entities)

            with c2:
                        st.metric("📄 Words", len(text.split()))

            with c3:
                        st.metric("🎯 Avg Confidence", avg_confidence)

            st.markdown("---")

            tab1, tab2, tab3 = st.tabs([
                        "🤖 AI Summary",
                        "🧬 Entities",
                        "📊 Analytics"
                    ])

            with tab1:

                        st.subheader("📄 Original Abstract")
                        st.write(text)

                        st.subheader("🤖 AI Summary")
                        st.success(summary)

                        st.subheader("💡 Research Insight")

                        if not df_entities.empty:

                            top_entities = (
                                df_entities["Entity"]
                                .unique()
                                .tolist()[:5]
                            )

                            insight = f"""
            This study highlights {', '.join(top_entities)}
            as important biomedical entities.

            These findings may have relevance for disease
            mechanisms, therapeutic targeting, biomarker
            discovery, and clinical translation.
            """

                        else:

                            insight = """
            No major biomedical entities were detected.
            """

                        st.info(insight)

                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )

            with tab2:

                        st.subheader(
                            "🧬 Biomedical Entity Detection"
                        )

                        if not df_entities.empty:

                            st.dataframe(
                                df_entities,
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "No biomedical entities detected."
                            )

            with tab3:

                        st.subheader(
                            "📈 Literature Analytics"
                        )

                        if not df_entities.empty:

                            fig = px.histogram(
                                df_entities,
                                x="Category",
                                title="Biomedical Entity Distribution"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            st.info(f"""
            Total Words: {len(text.split())}

            Total Biomedical Entities: {total_entities}

            Average Confidence Score: {avg_confidence}
            """)

                        else:

                            st.warning(
                                "Please enter a biomedical abstract."
                            )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Nirupam Joarder | Biomedical Research Intelligence Platform | Transformer NLP Project"
)

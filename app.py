import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
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

✅ Transformer Summarization

✅ Biomedical NER

✅ Research Insight Engine

✅ Literature Analytics

✅ Confidence Scoring

✅ Interactive Visualization
""")

    st.markdown("---")

    st.markdown("""
### Models

📄 BART Summarizer

facebook/bart-large-cnn

🧬 Biomedical NER

d4data/biomedical-ner-all
""")

    st.markdown("---")

    if st.button("📖 Load Sample Abstract"):

        st.session_state.sample_text = """
The EGFR T790M mutation is one of the major causes of acquired resistance to first-generation EGFR tyrosine kinase inhibitors in non-small cell lung cancer. Osimertinib has emerged as an effective therapeutic strategy targeting EGFR T790M-positive tumors.
"""

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_summarizer():

    return pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_summarizer()
@st.cache_resource
def load_ner_model():

    return pipeline(
        "token-classification",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple"
    )

ner_model = load_ner_model()
LABEL_MAP = {
    "Diagnostic_procedure": "Biomedical Marker",
    "Medication": "Therapeutic Agent",
    "Lab_value": "Clinical Parameter",
    "Sign_symptom": "Clinical Feature",
    "Coreference": "Research Entity"
}

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.title("🧬 Biomedical Research Intelligence Platform")

st.markdown("""
### Transformer-Powered Biomedical Literature Analysis

Using abstractive summarization and biomedical named entity recognition
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
if st.button("🔍 Analyze Literature"):

    if text.strip():

        with st.spinner("🧠 Running Transformer Model..."):

            # --------------------------
            # BART SUMMARY
            # --------------------------

            summary = summarizer(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
            )[0]["summary_text"]

            # --------------------------
            # BIOMEDICAL NER
            # --------------------------

            entities = ner_model(text)

            entity_data = []

            for ent in entities:

                category = LABEL_MAP.get(
                  ent["entity_group"],
                  ent["entity_group"]
                )

                entity_data.append({
                    "Entity": ent["word"],
                    "Category": category,
                    "Confidence": round(
                        float(ent["score"]),
                        3
                    )
                })

            df_entities = pd.DataFrame(entity_data)

            total_entities = len(df_entities)

            if not df_entities.empty:

                avg_confidence = round(
                    df_entities["Confidence"].mean(),
                    2
                )

            else:

                avg_confidence = 0

        # ---------------------------------
        # METRICS
        # ---------------------------------

        st.subheader("📊 Research Intelligence Metrics")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "🔬 Entities",
                total_entities
            )

        with c2:
            st.metric(
                "📄 Words",
                len(text.split())
            )

        with c3:
            st.metric(
                "🎯 Avg Confidence",
                avg_confidence
            )

        st.markdown("---")

        # ---------------------------------
        # TABS
        # ---------------------------------

        tab1, tab2, tab3 = st.tabs([
            "🤖 AI Summary",
            "🧬 Entities",
            "📊 Analytics"
        ])

        # ---------------------------------
        # SUMMARY TAB
        # ---------------------------------

        with tab1:

            st.subheader("📄 Original Abstract")

            st.write(text)

            st.subheader("🤖 AI Summary")

            st.success(summary)

            st.subheader("💡 Research Insight")

            if not df_entities.empty:

                top_entities = (
                    df_entities["Entity"]
                    .head(5)
                    .tolist()
                )

                insight = f"""
This study highlights {', '.join(top_entities)}
as important biomedical entities. These findings suggest
potential relevance to disease mechanisms, therapeutic
targeting, or clinical intervention.
"""

            else:

                insight = """
No significant biomedical entities were detected
within the supplied abstract.
"""

            st.info(insight)

            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

        # ---------------------------------
        # ENTITY TAB
        # ---------------------------------

        with tab2:

            st.subheader(
                "🧬 Biomedical Named Entity Recognition"
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

        # ---------------------------------
        # ANALYTICS TAB
        # ---------------------------------

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

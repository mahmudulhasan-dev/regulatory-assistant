"""
Minimal Streamlit interface over the existing pipeline.
No new retrieval or generation logic lives here - this only calls
generate.generate() and renders its result.
"""

import streamlit as st

import generate

st.set_page_config(page_title="Regulatory Assistant", page_icon="\U0001F4D8")

st.markdown(
    """
    <style>
    .answer-box {
        border-left: 4px solid #1e3a5f;
        background-color: #f7f9fb;
        padding: 1.25rem 1.5rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0 1.5rem 0;
    }
    .answer-label {
        color: #1e3a5f;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Regulatory Assistant")
st.caption(
    "Ask a question about AI regulation across five public documents: "
    "NIST AI RMF, NIST GenAI Profile, the EU AI Act, the UK AI white "
    "paper, and the Blueprint for an AI Bill of Rights. Answers are "
    "grounded in retrieved text and cite their source - the system "
    "declines rather than guessing when the corpus doesn't support an "
    "answer."
)

query = st.text_input("Your question", placeholder="e.g. What is a prohibited AI practice under the EU AI Act?")
submitted = st.button("Ask", type="primary")

if submitted and query.strip():
    with st.spinner("Retrieving and generating..."):
        result = generate.generate(query)

    top_score = result.retrieved[0]["score"] if result.retrieved else 0.0

    with st.expander("Retrieval detail (top match score, for transparency)"):
            st.caption(
                f"Top retrieval score: {top_score:.3f}. Scores below ~0.3 are "
                f"treated as out-of-corpus; scores in the 0.5-0.6 range can "
                f"indicate a topically related but unsupported question (see "
                f"README). This project's eval found some legitimate "
                f"questions score in this harder middle range when the exact "
                f"answer is a short definition embedded in longer text."
            )

    if not result.sufficient:
        st.warning(
            "This corpus doesn't contain enough information to answer that question.",
            icon="\u26a0\ufe0f",
        )
        with st.expander("Why did it decline?"):
            st.caption(result.decline_reason)
    else:
        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-label">Answer, based on the retrieved sources below</div>
                <div>{result.answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Sources")
        seen_indices = []
        for cite in result.citations:
            if cite["chunk_index"] not in seen_indices:
                seen_indices.append(cite["chunk_index"])
                c = cite["chunk"]
                label = c["section_label"] or f"page {c['page']}"
                with st.expander(f"[{cite['chunk_index']}] {c['doc_id']} \u2014 {label}"):
                    st.caption(f"Retrieval score: {c['score']:.3f}")
                    st.markdown(f"**Claim:** {cite['claim']}")
                    st.markdown("**Full source text:**")
                    st.write(c["text"])

elif submitted:
    st.info("Enter a question first.")
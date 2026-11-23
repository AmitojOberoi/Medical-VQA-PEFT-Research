import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Medical VQA | LLaVA-Med",
    page_icon="🩺",
    layout="wide"
)

st.title("Medical Visual Question Answering")
st.caption("LLaVA-Med 7B | Parameter-Efficient Fine-Tuning with LoRA")

st.info(
    "Research prototype. This application is intended for research and "
    "educational demonstrations, not clinical diagnosis."
)

with st.sidebar:
    st.header("Experiment Configuration")

    dataset = st.selectbox(
        "Select Dataset",
        ["VQA-RAD", "SLAKE", "PathVQA"]
    )

    adapter = st.selectbox(
        "Select Trained LoRA Adapter",
        ["VQA-RAD", "SLAKE", "PathVQA"]
    )

    st.divider()
    st.write("Base Model")
    st.code("microsoft/llava-med-v1.5-mistral-7b")

left, right = st.columns(2)

with left:
    st.subheader("Medical Image")
    uploaded_image = st.file_uploader(
        "Upload a medical image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded medical image", use_container_width=True)

with right:
    st.subheader("Ask a Medical Question")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What abnormality is visible in this X-ray?",
        height=130
    )

    generate = st.button(
        "Generate Answer",
        type="primary",
        use_container_width=True
    )

    if generate:
        if uploaded_image is None:
            st.warning("Please upload a medical image first.")
        elif not question.strip():
            st.warning("Please enter a medical question.")
        else:
            st.warning(
                "Inference backend is not connected yet. "
                "No model prediction has been generated."
            )

st.divider()
st.subheader("Model Output")

st.write("Waiting for inference backend connection.")

st.caption(
    "Selected dataset: " + dataset +
    " | Selected adapter: " + adapter
)

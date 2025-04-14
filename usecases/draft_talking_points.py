import streamlit as st
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain.chains import ConversationalRetrievalChain

def run(llm):
    st.subheader("🗣️ Draft Talking Points")

    # Initialize state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Step 1: Ask for the subject
    subject = st.text_input("Enter the subject of the talking points:")

    if not subject:
        st.warning("Please enter a subject to continue.")
        return

    # Step 2: Ask if they want to upload documents
    refer_documents = st.radio(
        "Do you want to upload reference documents for generating the talking points?",
        ["No", "Yes"],
        key="refer_docs_choice"
    )

    docs = None
    uploaded_files = []

    if refer_documents == "Yes":
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, XLSX)",
            type=["pdf", "docx", "txt", "xlsx"],
            accept_multiple_files=True,
            key="talking_points_upload"
        )
        if uploaded_files:
            with st.spinner("Processing uploaded documents..."):
                docs = load_documents(uploaded_files)
                st.session_state.vectorstore = create_vectorstore(docs)
                st.success(f"✅ {len(uploaded_files)} document(s) processed.")

    # Step 3: Generate button
    if st.button("Generate Talking Points"):
        with st.spinner("Generating talking points..."):

            if refer_documents == "Yes" and st.session_state.vectorstore:
                # Use documents + subject
                memory = get_memory()
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    memory=memory
                )
                query = f"Generate well-structured talking points on the subject of {subject}. Include both international and local content based on the provided documents."
                result = qa_chain.invoke({"question": query})
                talking_points = result["answer"]

            else:
                # Use LLM's own knowledge
                query = f"Generate well-structured talking points on the subject of {subject}. Include both international and local content."
                result = llm.invoke(query)
                talking_points = getattr(result, "content", str(result))

            formatted_answer = format_talking_points(talking_points)
            st.session_state.chat_history.append((subject, formatted_answer))

            # Show result
            st.markdown("### 💬 **Talking Points**\n\n" + formatted_answer)

            # Download option
            if st.button("📥 Download Talking Points as PDF"):
                export_answer_to_pdf(formatted_answer)

    # Step 4: Show last 3 chat entries
    if st.session_state.chat_history:
        st.markdown("### 📜 Recent Talking Points")
        for i, (q, a) in enumerate(st.session_state.chat_history[-3:]):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")

def format_talking_points(answer):
    formatted = answer.replace("\n", "\n\n")
    for numeral in ["I", "II", "III", "IV", "V", "VI", "VII"]:
        formatted = formatted.replace(f"{numeral}. ", f"\n\n### {numeral}. **")
    formatted = formatted.replace("- ", "\n- ")
    return formatted

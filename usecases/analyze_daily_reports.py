# usecases/analyze_daily_reports.py
import streamlit as st
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain.chains import ConversationalRetrievalChain

def run(llm):
    st.subheader("🗂️ Analyze Daily Reports")

    # Step 1: Ask user what they want to do
    analysis_type = st.radio(
        "What do you want to analyze?",
        options=["", "Analyze a single report", "Analyze multiple reports together"],
        index=0,
        format_func=lambda x: "— Select an option —" if x == "" else x
    )

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    docs = None

    # Step 2: Show file uploader only after user makes a choice
    if analysis_type:
        if analysis_type == "Analyze a single report":
            uploaded_file = st.file_uploader(
                "📄 Upload a single daily report",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=False,
                key="single_report"
            )
            if uploaded_file:
                docs = load_documents([uploaded_file])

        elif analysis_type == "Analyze multiple reports together":
            with st.expander("📁 Upload multiple daily reports"):
                st.markdown("""
                Upload **two or more daily reports** to:
                - Compare 📊
                - Contrast ⚖️
                - Consolidate 📚
                - Draw conclusions 🧠

                _Accepted formats: `.txt`, `.pdf`, `.docx`_
                """)
                uploaded_files = st.file_uploader(
                    "Choose your report files",
                    type=["txt", "pdf", "docx"],
                    accept_multiple_files=True,
                    key="multi_reports"
                )
                if uploaded_files:
                    st.markdown("### ✅ Files uploaded:")
                    for file in uploaded_files:
                        st.write(f"- {file.name}")
                    docs = load_documents(uploaded_files)

    # Step 3: Create vectorstore
    if docs:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = create_vectorstore(docs)
            st.success("✅ Documents processed!")

    # Step 4: Show question input + results
    if st.session_state.vectorstore:
        memory = get_memory()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )

        query_label = (
            "What would you like to know about the report?"
            if analysis_type == "Analyze a single report"
            else "What would you like to do with the multiple reports?"
        )
        query = st.text_input(query_label)

        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"question": query})
                answer = result["answer"]
                st.session_state.chat_history.append((query, answer))

                st.markdown("### 💬 Answer")
                st.write(answer)

                st.markdown("### 📜 Chat History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**A{i+1}:** {a}")

                if st.button("📥 Download Latest Answer as PDF"):
                    export_answer_to_pdf(answer)

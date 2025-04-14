import streamlit as st
import pandas as pd
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain.chains import ConversationalRetrievalChain


def run(llm):
    st.subheader("📝 Write Reports")

    # Initialize session state variables
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Step 1: Upload files (mandatory)
    uploaded_files = st.file_uploader(
        "📂 Upload reference documents (PDF, Word, Excel, etc.)",
        type=["pdf", "docx", "txt", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Please upload at least one document to proceed.")
        return

    # Step 2: Load and process uploaded documents
    with st.spinner("Processing uploaded documents..."):
        docs = load_documents(uploaded_files)
        st.session_state.vectorstore = create_vectorstore(docs)
    st.success("✅ Documents processed and ready for report writing!")

    # Step 3: Get user input for the report title
    report_title = st.text_input("🧾 What is the title or subject of the report?")
    if not report_title:
        st.warning("Please enter a title for the report.")
        return

    # Step 4: Optional - Structured Table of Contents via editable table
    st.markdown("### 📑 Optional: Enter Table of Contents (Up to 3 levels)")

    # Default DataFrame for sections and subsections
    default_df = pd.DataFrame({
        "Level": [1, 2, 3, 2, 3],
        "Section Number": ["1", "1.1", "1.1.1", "2", "2.1"],
        "Title": ["Introduction", "Scope", "Scope Details", "Literature Review", "Review Overview"]
    })

    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key="toc_editor"
    )

    # Build TOC string from the DataFrame
    toc_input = ""
    for _, row in edited_df.iterrows():
        level = int(row["Level"])
        section_number = row["Section Number"]
        title = row["Title"]
        
        # Format TOC string based on the level of the section
        toc_input += f"{' ' * (level - 1) * 2}{section_number}. {title}\n"

    # Step 5: Generate report when button is clicked
    if st.button("✍️ Generate Report"):
        memory = get_memory()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )

        # Create query prompt with or without TOC input
        if toc_input:
            query = f"""Write a comprehensive and structured report on the topic: "{report_title}".
Include insights from the uploaded documents and current knowledge. 
Use the following table of contents with three levels:
{toc_input}"""
        else:
            query = f"""Write a comprehensive and structured report on the topic: "{report_title}".
Include insights from the uploaded documents and current knowledge."""

        with st.spinner("Generating report..."):
            result = qa_chain.invoke({"question": query})
            report_content = result["answer"]

        st.markdown("### 📄 **Generated Report**")
        st.markdown(report_content)

        if st.button("📥 Download Report as PDF"):
            export_answer_to_pdf(report_content)

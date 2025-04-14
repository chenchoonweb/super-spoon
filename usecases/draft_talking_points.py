
# usecases/draft_talking_points.py
import streamlit as st
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain.chains import ConversationalRetrievalChain

def run(llm):
    st.subheader("🗣️ Draft Talking Points")

    # Step 1: Ask user what they want to do
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    subject = st.text_input("What is the subject of the talking point?")
    if not subject:
        st.warning("Please enter a subject for the talking point.")
        return

    # Step 2: Ask if the user wants to refer to documents for drafting talking points
    refer_documents = st.radio(
        "Would you like to refer to one or more documents (PDF, Word, Excel, etc.) for drafting the talking points?",
        ["No", "Yes"]
    )

    docs = None
    if refer_documents == "Yes":
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt", "xlsx"],
            accept_multiple_files=True
        )
        if uploaded_files:
            docs = load_documents(uploaded_files)

    # Step 3: Process and create vectorstore
    if docs:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = create_vectorstore(docs)
            st.success("✅ Documents processed!")

    # Step 4: If vectorstore is ready, allow question input for generating talking points
    if st.session_state.vectorstore:
        memory = get_memory()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )

        # Generate talking points based on the subject entered by the user
        if subject:
            query = f"Generate well-structured talking points on the subject of {subject}. Include both international and local content based on the provided documents."
            with st.spinner("Generating talking points..."):
                result = qa_chain.invoke({"question": query})
                talking_points = result["answer"]
                st.session_state.chat_history.append((query, talking_points))

                # Format the answer dynamically based on LLM's response
                formatted_answer = format_talking_points(talking_points)

                st.markdown("### 💬 **Talking Points**\n\n" + formatted_answer)

                # Option to download the answer as PDF
                if st.button("📥 Download Talking Points as PDF"):
                    export_answer_to_pdf(formatted_answer)

                st.markdown("### 📜 Chat History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**A{i+1}:** {a}")

def format_talking_points(answer):
    """
    Function to format the raw talking points into a readable structure with headings, bullet points, and sections.
    This function ensures that the content provided by the LLM is well-formatted.
    """
    # Add line breaks for better readability
    formatted_answer = answer.replace("\n", "\n\n")
    
    # Make section headers bold and ensure proper structure
    formatted_answer = formatted_answer.replace("I. ", "\n\n### I. **").replace("II. ", "\n\n### II. **") \
        .replace("III. ", "\n\n### III. **").replace("IV. ", "\n\n### IV. **").replace("V. ", "\n\n### V. **") \
        .replace("VI. ", "\n\n### VI. **").replace("VII. ", "\n\n### VII. **")
    
    # Add bold formatting for section titles
    formatted_answer = formatted_answer.replace("**Understanding Trade War**", "### **Understanding Trade War**") \
        .replace("**Causes of Trade Wars**", "### **Causes of Trade Wars**") \
        .replace("**Examples of Trade Wars**", "### **Examples of Trade Wars**") \
        .replace("**Impact of Trade Wars**", "### **Impact of Trade Wars**") \
        .replace("**Trade Wars and the Local Context**", "### **Trade Wars and the Local Context**") \
        .replace("**Mitigating the Impact of Trade Wars**", "### **Mitigating the Impact of Trade Wars**") \
        .replace("**The Future of Trade Wars**", "### **The Future of Trade Wars**")
    
    # Add bullet points for better structure if needed
    formatted_answer = formatted_answer.replace("- ", "\n- ")

    return formatted_answer

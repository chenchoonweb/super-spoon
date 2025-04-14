import streamlit as st
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain.chains import ConversationalRetrievalChain

def run(llm):
    st.subheader("🗣️ Draft Talking Points")

    # Step 1: Ask user for the subject of the talking points
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clearly ask for the subject
    subject = st.text_input("What is the subject of the talking points you want to generate?")

    # If no subject is provided, give a warning
    if not subject:
        st.warning("Please enter a subject for the talking points to continue.")
        return  # Prevent moving forward until a subject is provided

    # Step 2: Ask if the user wants to upload local reference documents for talking points
    refer_documents = st.radio(
        "Do you want to upload any local reference documents (PDF, Word, Excel, etc.) for creating the talking points?",
        ["No", "Yes"]
    )

    docs = None
    if refer_documents == "Yes":
        # Step 3: File upload for documents if "Yes" is selected
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt", "xlsx"],
            accept_multiple_files=True
        )
        
        # If documents are uploaded, process them
        if uploaded_files:
            docs = load_documents(uploaded_files)
            st.success(f"Successfully uploaded {len(uploaded_files)} document(s).")
        else:
            st.warning("Please upload at least one document to proceed with reference material.")

    # Step 4: Process documents and create vectorstore if documents are uploaded
    if docs:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = create_vectorstore(docs)
            st.success("✅ Documents processed and ready for talking points generation!")

    # Step 5: If no documents are uploaded, use LLM's knowledge to generate talking points
    if not docs:
        with st.spinner("Generating talking points from LLM's knowledge..."):
            result = llm(subject)  # This part generates talking points from LLM
            talking_points = result['text']  # Assume LLM returns the talking points in the 'text' field

            # Display talking points
            st.markdown(f"### 💬 **Generated Talking Points on {subject}**\n\n{talking_points}")

    # Step 6: If vectorstore is created (i.e., documents uploaded), generate talking points
    elif st.session_state.vectorstore:
        # Use Conversational Retrieval Chain to generate points based on subject
        memory = get_memory()
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )

        query = f"Generate well-structured talking points on the subject of {subject}. Include both international and local content based on the provided documents."
        
        with st.spinner("Generating talking points..."):
            result = qa_chain.invoke({"question": query})
            talking_points = result["answer"]
            st.session_state.chat_history.append((query, talking_points))

            # Format and display talking points
            formatted_answer = format_talking_points(talking_points)
            st.markdown(f"### 💬 **Talking Points**\n\n{formatted_answer}")

            # Option to download the talking points as a PDF
            if st.button("📥 Download Talking Points as PDF"):
                export_answer_to_pdf(formatted_answer)

    # Step 7: Display chat history with a limit (showing the last 5 exchanges)
    st.markdown("### 📜 Chat History")
    max_history = 5
    for i, (q, a) in enumerate(st.session_state.chat_history[-max_history:]):
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


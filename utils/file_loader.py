# utils/file_loader.py
import tempfile
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        suffix = f".{uploaded_file.type.split('/')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if suffix == ".txt":
            loader = TextLoader(tmp_file_path)
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        documents.extend(loader.load())
    return documents


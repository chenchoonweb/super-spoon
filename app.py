import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from utils.file_loader import load_documents
from utils.vectorstore import create_vectorstore
from utils.memory import get_memory
from utils.pdf_exporter import export_answer_to_pdf
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import importlib


load_dotenv()

st.set_page_config(page_title="📄 AI Document Agent", layout="centered")

# Load and display logo
logo = Image.open("assets/logo.jpeg")
st.image(logo, width=150)

st.write("A test AI Tool being developed by Cabinet Secretariat officials for inhouse use")

# Use case selection
use_case = st.selectbox(
    "What would you like to use the AI agent for?",
    [
        "Analyze Policy or Laws",
        "Analyze Daily Reports",
        "Analyze Financials",
        "Write Reports",
        "Draft Speech",
        "Draft Talking Points",
        "Summarize Reports"
    ]
)
st.write(f"Selected option: **{use_case}**")


# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load selected use case module dynamically
# Always load and run the selected use case module
llm = ChatOpenAI(model="gpt-4", temperature=0)

module_name = use_case.lower().replace(" ", "_").replace("-", "_")
try:
    use_case_module = importlib.import_module(f"usecases.{module_name}")
    print("✅ Module imported successfully:", use_case_module)
    use_case_module.run(llm)
    print("🚀 run() executed")
except ModuleNotFoundError as e:
    st.error(f"Use case module not found: {e}")
    print("❌ ModuleNotFoundError:", e)
except AttributeError as e:
    st.error(f"Use case module missing `run()` function: {e}")
    print("❌ AttributeError:", e)
except Exception as e:
    st.error(f"Error running use case: {e}")
    print("❌ General error:", e)

# Sidebar and footer
st.sidebar.markdown("### 👤 About")
st.sidebar.markdown("""
**Org:** Cabinet Secretariat  
**Version:** 1.0.0
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Need help? [Contact us](mailto:chencho@cabinet.gov.bt)")

st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; color: grey;">
        © 2025 Cabinet Secretariat
    </div>
    """,
    unsafe_allow_html=True
)

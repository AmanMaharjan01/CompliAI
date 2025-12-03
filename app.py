import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required API keys before importing modules
def check_api_keys():
    """Check if required API keys are configured"""
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    missing_keys = []
    warnings = []
    
    # Check embedding provider requirements
    if embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY (for OpenAI embeddings)")
    elif embedding_provider == "gemini":
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            missing_keys.append("GOOGLE_API_KEY (for Gemini embeddings)")
        elif len(gemini_key.strip()) < 20:
            warnings.append(f"GOOGLE_API_KEY appears too short ({len(gemini_key)} chars)")
    
    # Check LLM provider requirements
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY (for OpenAI LLM)")
    elif llm_provider == "gemini":
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            missing_keys.append("GOOGLE_API_KEY (for Gemini LLM)")
        elif len(gemini_key.strip()) < 20:
            warnings.append(f"GOOGLE_API_KEY appears too short ({len(gemini_key)} chars)")
    
    return missing_keys, warnings, embedding_provider, llm_provider

# Check API keys first
missing_keys, warnings, embedding_provider, llm_provider = check_api_keys()

if missing_keys or warnings:
    st.set_page_config(
        page_title="CompliAI - Setup Required",
        page_icon="⚠️",
        layout="wide"
    )
    
    if missing_keys:
        st.error("### ⚠️ Configuration Required")
        st.markdown(f"""
        **Current Configuration:**
        - Embeddings: `{embedding_provider.upper()}`
        - LLM: `{llm_provider.upper()}`
        
        **Missing API Keys:**
        """)
        
        for key in missing_keys:
            st.markdown(f"- ❌ `{key}`")
    
    if warnings:
        st.warning("### ⚠️ Potential Issues")
        for warning in warnings:
            st.markdown(f"- ⚠️ {warning}")
        
        st.info("""
        **How to fix Gemini API Key issues:**
        
        1. Go to: https://makersuite.google.com/app/apikey
        2. Click "Create API Key"
        3. Copy the ENTIRE key (usually starts with `AIza...`)
        4. Edit your `.env` file:
           ```
           GOOGLE_API_KEY=AIzaSyD...your_full_key_here
           ```
        5. Important: NO quotes, NO spaces before/after the key
        6. Restart the application
        
        **Example of CORRECT .env format:**
        ```
        GOOGLE_API_KEY=AIzaSyDXXXXXXXXXXXXXXXXXXXXXXXX
        ```
        
        **Example of WRONG format:**
        ```
        GOOGLE_API_KEY="AIzaSyD..."  ← Don't use quotes
        GOOGLE_API_KEY= AIzaSyD...   ← Don't add space after =
        ```
        """)
    
    st.info("""
    ### 🆓 Want to use CompliAI for FREE with no API issues?
    
    Edit your `.env` file:
    ```
    EMBEDDING_PROVIDER=huggingface
    LLM_PROVIDER=gemini
    GOOGLE_API_KEY=your_gemini_key
    ```
    
    Or for **completely offline** (no API calls at all):
    ```
    EMBEDDING_PROVIDER=huggingface
    LLM_PROVIDER=ollama
    ```
    """)
    
    st.stop()

# Import modules after API key check
from src.rag.query_engine import QueryEngine
from src.ingestion.document_processor import DocumentProcessor
from src.utils.vector_store import VectorStoreManager

# Page configuration
st.set_page_config(
    page_title="CompliAI - Policy Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .compliai-logo {
        font-weight: bold;
        color: #1f77b4;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with error handling
if 'query_engine' not in st.session_state:
    try:
        # Show initialization message
        with st.spinner(f"Initializing CompliAI with {embedding_provider} embeddings..."):
            if embedding_provider == "huggingface":
                st.info("📥 First time setup: Downloading embedding model (this may take 1-2 minutes)...")
            
            st.session_state.query_engine = QueryEngine()
            
        logger.info(f"Initialized with Embeddings: {embedding_provider}, LLM: {llm_provider}")
        
    except Exception as e:
        error_msg = str(e)
        
        # Check for Gemini API key errors
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            st.error("### 🔑 Invalid Gemini API Key")
            st.markdown("""
            Your Gemini API key is not valid or not working.
            
            **Steps to fix:**
            
            1. **Get a NEW API key:**
               - Visit: https://makersuite.google.com/app/apikey
               - Click "Create API Key"
               - Select your Google Cloud project (or create one)
               - Copy the ENTIRE key (starts with `AIza`)
            
            2. **Update your .env file:**
               ```
               GOOGLE_API_KEY=AIzaSyD_your_full_key_here_no_quotes
               ```
            
            3. **Common mistakes to avoid:**
               - ❌ Don't use quotes: `"AIza..."`
               - ❌ Don't add spaces: `GOOGLE_API_KEY= AIza`
               - ❌ Don't copy partial key
               - ✅ Copy the full key without any extra characters
            
            4. **Verify your key format:**
               - Should start with: `AIza`
               - Should be 39-40 characters long
               - Should contain letters, numbers, dashes, underscores
            
            5. **Restart the application**
            
            **Alternative: Use completely free offline mode:**
            ```
            EMBEDDING_PROVIDER=huggingface
            LLM_PROVIDER=ollama
            ```
            (No API keys needed!)
            """)
        
        # Check for specific errors
        if "sentence-transformers" in error_msg or "sentence_transformers" in error_msg:
            st.error("### 📦 Missing Package: sentence-transformers")
            st.markdown("""
            **HuggingFace embeddings require the sentence-transformers package.**
            
            **To install:**
            ```bash
            pip3 install sentence-transformers
            ```
            
            **Or switch to Gemini embeddings (requires API key):**
            
            Edit your `.env` file:
            ```
            EMBEDDING_PROVIDER=gemini
            GOOGLE_API_KEY=your_free_gemini_key
            ```
            
            Then restart the application.
            """)
            
        elif "insufficient_quota" in error_msg or "429" in error_msg:
            st.error("### 💳 OpenAI API Quota Exceeded")
            st.markdown("""
            Your OpenAI API account has exceeded its quota.
            
            **FREE Alternative - Switch to HuggingFace:**
            ```bash
            # Install
            pip3 install sentence-transformers
            
            # Edit .env
            EMBEDDING_PROVIDER=huggingface
            LLM_PROVIDER=gemini
            GOOGLE_API_KEY=your_free_gemini_key
            ```
            """)
        else:
            st.error(f"❌ Failed to initialize Query Engine: {str(e)}")
            st.info("Please check your configuration and try again.")
        
        # Show detailed error in expander
        with st.expander("🔍 Technical Details"):
            st.code(error_msg)
        
        st.stop()

if 'doc_processor' not in st.session_state:
    try:
        st.session_state.doc_processor = DocumentProcessor()
    except Exception as e:
        st.error(f"❌ Failed to initialize Document Processor: {str(e)}")
        st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 <span class="compliai-logo">CompliAI</span> - Policy & Compliance Assistant</h1>', unsafe_allow_html=True)
    
    # Show current configuration
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        if embedding_provider == "huggingface":
            st.success(f"📊 Embeddings: **{embedding_provider.upper()}** (FREE)")
        else:
            st.info(f"📊 Embeddings: **{embedding_provider.upper()}**")
    
    with col_info2:
        if llm_provider == "ollama":
            st.success(f"🤖 LLM: **{llm_provider.upper()}** (FREE)")
        elif llm_provider == "gemini":
            # Show simplified model name
            display_model = gemini_model.replace("-latest", "").replace("gemini-", "")
            st.success(f"🤖 LLM: **Gemini {display_model.upper()}** (FREE)")
        else:
            st.info(f"🤖 LLM: **{llm_provider.upper()}**")
    
    with col_info3:
        if llm_provider == "gemini":
            if "flash" in gemini_model.lower():
                st.info("⚡ 1500 req/day • 15 RPM")
            elif "pro" in gemini_model.lower():
                st.warning("🧠 50 req/day • 2 RPM")
    
    if embedding_provider == "huggingface":
        st.success("✅ Using FREE local embeddings - NO API calls for document processing!")
    
    if llm_provider == "gemini" and "flash" in gemini_model.lower():
        st.success("✅ Using Gemini 1.5 Flash - Best FREE tier option with generous quotas!")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload Policy Document",
            type=['pdf', 'docx', 'txt', 'md'],
            help="Upload company policy documents"
        )
        
        if uploaded_file:
            st.subheader("Document Metadata")
            
            department = st.selectbox(
                "Department",
                ["HR", "IT", "Legal", "Operations", "Finance", "Other"]
            )
            
            policy_type = st.text_input("Policy Type", "General Policy")
            
            effective_date = st.date_input("Effective Date")
            
            description = st.text_area("Description (Optional)")
            
            if st.button("📥 Process Document", type="primary"):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message, progress):
                    status_text.text(message)
                    progress_bar.progress(progress)
                
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_path = upload_dir / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document with progress
                        metadata = {
                            "department": department,
                            "policy_type": policy_type,
                            "effective_date": str(effective_date),
                            "description": description
                        }
                        
                        result = st.session_state.doc_processor.ingest_document(
                            file_path=str(file_path),
                            metadata=metadata,
                            progress_callback=update_progress  # NEW
                        )
                        
                        if result['status'] == 'success':
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"📊 Created {result['num_chunks']} chunks")
                            
                            # Show API usage estimate
                            st.caption(f"💰 Estimated API calls: ~{result['num_chunks']} embeddings")
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                    except Exception as e:
                        st.error(f"❌ Failed to process document: {str(e)}")
                    finally:
                        progress_bar.empty()
                        status_text.empty()

        st.markdown("---")
        
        # Vector store stats
        st.subheader("📊 CompliAI Stats")
        try:
            vector_store = VectorStoreManager()
            stats = vector_store.get_stats()
            
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Queries Processed", st.session_state.query_count)
        except Exception as e:
            st.warning("⚠️ Unable to load stats")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        use_reranking = st.checkbox("Use Cohere Reranking", value=True, 
                                    help="Requires COHERE_API_KEY in .env")
        k_documents = st.slider("Retrieved Documents", 1, 20, 5)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Query input
        question = st.text_input(
            "Enter your policy question:",
            placeholder="e.g., What is the PTO policy?",
            label_visibility="collapsed"
        )
        
        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            submit_query = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Sample questions
        st.caption("**Sample Questions:**")
        sample_questions = [
            "What is the remote work policy?",
            "How do I request parental leave?",
            "What is the password policy?",
            "What are the confidentiality requirements?",
            "How do I report a security incident?"
        ]
        
        selected_sample = st.selectbox(
            "Or choose a sample question:",
            [""] + sample_questions,
            label_visibility="collapsed"
        )
        
        if selected_sample:
            question = selected_sample
            submit_query = True
        
        # Process query
        if submit_query and question:
            with st.spinner("Searching policies..."):
                try:
                    # Prepare chat history
                    chat_history_str = "\n".join([
                        f"Q: {item['question']}\nA: {item['answer']}"
                        for item in st.session_state.chat_history[-3:]  # Last 3 exchanges
                    ])
                    
                    # Query
                    result = st.session_state.query_engine.query(
                        question=question,
                        chat_history=chat_history_str if chat_history_str else None,
                        k=k_documents
                    )
                    
                    # Update history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"]["summary"],
                        "full_result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.query_count += 1
                    
                    # Display result
                    st.markdown("### 📋 Answer")
                    
                    # Summary
                    st.markdown(f"**Summary:** {result['answer']['summary']}")
                    
                    # Detailed answer
                    with st.expander("📖 Detailed Answer", expanded=True):
                        st.markdown(result['answer']['detailed_answer'])
                    
                    # Confidence
                    confidence = result['answer']['confidence']['level']
                    confidence_class = f"confidence-{confidence.lower()}"
                    
                    st.markdown(
                        f"**Confidence:** <span class='{confidence_class}'>{confidence}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(result['answer']['confidence']['reasoning'])
                    
                    # Policy references
                    if result['answer']['policy_references']:
                        st.markdown("### 📚 Policy References")
                        for i, ref in enumerate(result['answer']['policy_references'], 1):
                            with st.container():
                                st.markdown(f"**Reference {i}:**")
                                st.markdown(f"- **Source:** {ref.get('source', 'N/A')}")
                                if 'quote' in ref:
                                    st.markdown(f"- **Quote:** _{ref['quote']}_")
                                st.markdown("---")
                    
                    # Action items
                    if result['answer'].get('action_items'):
                        st.markdown("### ✅ Action Items")
                        for item in result['answer']['action_items']:
                            st.markdown(f"- {item}")
                    
                    # Related topics
                    if result['answer'].get('related_topics'):
                        st.markdown("### 🔗 Related Topics")
                        for topic in result['answer']['related_topics']:
                            st.markdown(f"- {topic}")
                    
                    # Metadata
                    with st.expander("🔍 Query Metadata"):
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric("Sources Used", result['metadata']['num_sources'])
                        
                        with col_m2:
                            st.metric(
                                "Response Time",
                                f"{result['metadata']['performance']['total_time_ms']:.0f} ms"
                            )
                        
                        with col_m3:
                            grounded = "✅ Yes" if result['metadata']['quality']['is_grounded'] else "⚠️ No"
                            st.metric("Grounded", grounded)
                        
                        # Sources details
                        st.markdown("**Sources:**")
                        for source in result['metadata']['sources']:
                            st.text(f"• {source.get('document', 'N/A')} (Page {source.get('page', 'N/A')})")
                
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
                    st.info("Please check your configuration and try again.")
    
    with col2:
        st.subheader("📜 Chat History")
        
        if st.session_state.chat_history:
            for i, item in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
                with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {item['question'][:50]}..."):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer']}")
                    st.caption(f"Time: {item['timestamp']}")
        else:
            st.info("No queries yet. Ask a question to get started!")
        
        # Export history
        if st.session_state.chat_history:
            if st.button("💾 Export History"):
                history_json = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    "Download JSON",
                    history_json,
                    "chat_history.json",
                    "application/json"
                )


if __name__ == "__main__":
    main()

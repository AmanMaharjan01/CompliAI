import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import json

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

# Initialize session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = QueryEngine()

if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 <span class="compliai-logo">CompliAI</span> - Policy & Compliance Assistant</h1>', unsafe_allow_html=True)
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
                with st.spinner("Processing document..."):
                    # Save uploaded file
                    upload_dir = Path("data/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process document
                    metadata = {
                        "department": department,
                        "policy_type": policy_type,
                        "effective_date": str(effective_date),
                        "description": description
                    }
                    
                    result = st.session_state.doc_processor.ingest_document(
                        file_path=str(file_path),
                        metadata=metadata
                    )
                    
                    if result['status'] == 'success':
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"Created {result['num_chunks']} chunks")
                    else:
                        st.error(f"❌ Error: {result.get('error')}")
        
        st.markdown("---")
        
        # Vector store stats
        st.subheader("📊 CompliAI Stats")
        vector_store = VectorStoreManager()
        stats = vector_store.get_stats()
        
        st.metric("Total Chunks", stats.get('total_chunks', 0))
        st.metric("Queries Processed", st.session_state.query_count)
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        use_reranking = st.checkbox("Use Cohere Reranking", value=True)
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

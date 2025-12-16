import streamlit as st
import os
import time
from PIL import Image

# Internal Script Imports
from scripts.architect import RAGArchitect
from scripts.vision_client import analyze_image
from scripts.retriever import retrieve_poems, get_embedding
from scripts.generator import generate_poem

# Safe Import for Audio Only (Visualizer Removed for Stability)
try:
    from scripts.audio import AudioEngine
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# --- Config ---
st.set_page_config(layout="wide", page_title="Poetic Camera")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .stButton>button { background-color: #238636; color: white; border-radius: 5px; height: 3em; font-family: monospace; }
    h1, h2, h3 { font-family: 'Courier New', Courier, monospace; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHING FUNCTIONS ---
@st.cache_data(show_spinner=False)
def run_vision_cached(image_file):
    """
    Caches the expensive vision analysis call so reruns (like changing sliders)
    don't re-trigger the Llama Vision model.
    """
    try:
        # Save to temp file for the vision client to read
        with open("temp_input.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        return analyze_image("temp_input.jpg")
    except Exception as e:
        return f"Error: {e}"

# --- Session State Initialization ---
# REMOVED: 'critique' from keys
keys = ['narrative', 'retrieved_items', 'generated_poem', 'audio_bytes', 'last_upload_id']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# Initialize Architect (Lazy Loading)
if 'rag_architect' not in st.session_state:
    st.session_state.rag_architect = RAGArchitect()

# ==========================================
# 1. SIDEBAR (Controls Only)
# ==========================================
with st.sidebar:

    st.header("Poet Persona")
    # Mapping UI Name -> Pinecone Namespace
    poet_map = {
        "Emily Dickinson": "dickinson",
        "Percy Bysshe Shelley": "shelley",
        "Walt Whitman": "whitman" 
    }
    
    selected_poet_name = st.selectbox(
        "Choose your Muse",
        options=list(poet_map.keys()),
        index=0 # Default to Dickinson
    )
    
    # Get the actual namespace string (e.g., 'shelley')
    target_namespace = poet_map[selected_poet_name]

    st.header("Input Configuration")
    
    # Select Mode
    input_method = st.radio(
        "Source", 
        ["Upload", "Camera"], 
        label_visibility="collapsed",
        key="input_mode" 
    )
    
    # Upload Logic (Stays in Sidebar because it doesn't need width)
    sidebar_upload = None
    if input_method == "Upload":
        sidebar_upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    if st.button("System Reset"):
        st.cache_data.clear()
        for k in keys: st.session_state[k] = None
        st.rerun()

# ==========================================
# MAIN LOGIC
# ==========================================
st.title("Poetic Camera")
# UPDATED: Reflected new mode
st.caption("System Status: Online | Mode: Production RAG (Cohere Powered)")

# --- CAMERA HANDLING ---
image_source = None

if input_method == "Upload":
    image_source = sidebar_upload
elif input_method == "Camera":
    # --- CAMERA IN MAIN AREA ---
    with st.expander("Open Viewfinder", expanded=(st.session_state.last_upload_id is None)):
        camera_shot = st.camera_input("Capture Scene")
        if camera_shot:
            image_source = camera_shot

# --- PROCESSING PIPELINE ---
if image_source:
    
    # Check for new file to reset state
    file_id = f"{image_source.name}_{image_source.size}"
    if st.session_state.last_upload_id != file_id:
        st.session_state.narrative = None
        st.session_state.retrieved_items = None
        st.session_state.generated_poem = None
        st.session_state.audio_bytes = None 
        # REMOVED: st.session_state.critique cleanup
        st.session_state.last_upload_id = file_id

    # Layout: 3 Columns
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    # --- CARD 1: VISUAL INGESTION ---
    with col1:
        with st.container(border=True):
            st.subheader("I. Ingestion")
            
            # Display Image
            st.image(image_source)
            
            # Metadata
            img = Image.open(image_source)
            st.caption(f"Res: {img.size[0]} x {img.size[1]} px")

    # --- CARD 2: INTERNAL MONOLOGUE ---
    with col2:
        with st.container(border=True):
            st.subheader("II. Processing")
            
            # 1. Vision Analysis
            if not st.session_state.narrative:
                with st.status("[SYSTEM] Initializing Vision Pipeline...", expanded=True) as s:
                    st.write("Task: Image Analysis (Llama 3.2 Vision)")
                    
                    # Capture the result
                    result = run_vision_cached(image_source) 
                    st.session_state.narrative = result
                    
                    s.update(label="[SYSTEM] Vision Analysis: Complete", state="complete", expanded=False)
            
            # --- ERROR HANDLING & RETRIEVAL ---
            if not st.session_state.narrative:
                st.error("Vision Analysis returned no data. Check logs.")
            
            # Check for explicit error
            elif st.session_state.narrative.startswith("ERROR:") or "Error:" in st.session_state.narrative:
                st.error(f"Pipeline Failed: {st.session_state.narrative}")
                st.stop() 
            
            # Proceed if valid
            else:
                st.info(f"**Narrative:** {st.session_state.narrative}")

                # 2. Memory Retrieval
                if not st.session_state.retrieved_items:
                    with st.status("[SYSTEM] Architect: Selecting References...", expanded=True) as s:
                    
                        st.write(f"1. Retrieving top 15 candidates from namespace: {target_namespace}...")
                        
                        raw_candidates = retrieve_poems(
                            st.session_state.narrative, 
                            top_k=15, 
                            namespace=target_namespace 
                        )
                    
                        # UPDATED: Text to reflect Cohere usage
                        st.write("2. Cohere Architect: Semantic Re-ranking...")
                        selected_candidates = st.session_state.rag_architect.select_best_candidates(
                            st.session_state.narrative, 
                            raw_candidates,
                            top_k=3
                        )
                    
                        st.session_state.retrieved_items = selected_candidates
                        s.update(label="[SYSTEM] Reference Selection Complete", state="complete", expanded=False)
            
                # Persistent State (If data exists)
                else:
                    with st.status("[SYSTEM] Memory Active", state="complete", expanded=False):
                        st.write("1. Vector Search: Complete")
                        st.write("2. Cohere Reranking: Complete")

    # --- CARD 3: GENERATIVE INFERENCE ---
    with col3:
        with st.container(border=True):
            st.subheader("III. Output")
            
            # Initialize temperature to a default to prevent 'UnboundLocalError'
            temperature = 0.5 
            
            if st.session_state.retrieved_items:
                
                st.markdown("#### Parameters")
                temperature = st.slider("Model creative freedom", 0.1, 1.0, 0.5)
                
                with st.expander("Context Data"):
                    for i, m in enumerate(st.session_state.retrieved_items):
                        meta = m.get('metadata', {})
                        raw_title = meta.get('title', f"{i+1}")
                        clean_text = meta.get('text', "No text.").strip()
                        
                        # Optional: Display Relevance Score if available (added by Cohere)
                        score_display = ""
                        if 'relevance_score' in m:
                            score_display = f" [Rel: {m['relevance_score']:.3f}]"

                        # Clean up title formatting
                        clean_title = raw_title
                        if "poem poem" in clean_title.lower():
                            clean_title = clean_title.lower().replace("poem poem", "Poem").title()
                        clean_title = clean_title.replace("_", " ").title()

                        st.markdown(f"**{clean_title}**{score_display}")
                        st.caption(f"{clean_text[:350] + '...' if len(clean_text) > 350 else clean_text}") 
                        st.divider()
                
                # --- BUTTON LOGIC ---
                st.markdown("---")
                if st.button("Generate poem with voice", type="primary", use_container_width=True):
                
                    # 1. TEXT GENERATION
                    with st.status("Drafting Poem...", expanded=True) as status:
                        st.write(f"Task: Text Inference (Style: {selected_poet_name})")
                    
                        st.session_state.generated_poem = generate_poem(
                            st.session_state.narrative,
                            st.session_state.retrieved_items,
                            poet_name=selected_poet_name,
                            temperature=temperature
                        )
                        
                        # REMOVED: Critique Logic Block
                        
                        status.update(label="Poem Drafted!", state="complete", expanded=False)

                    # 2. RENDER POEM
                    if st.session_state.generated_poem:
                        clean_poem = st.session_state.generated_poem.replace("- ", "— ")
                    
                        st.markdown(
                            f"<div style='text-align: center; font-style: italic; padding: 10px; font-family: serif; white-space: pre-wrap;'>{clean_poem}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # REMOVED: Critique Scorecard Display

                    # 3. AUDIO GENERATION
                    if AUDIO_AVAILABLE and st.session_state.generated_poem:
                        audio_placeholder = st.empty()
                        with audio_placeholder.status("Synthesizing Audio...", expanded=False) as audio_status:
                            audio = AudioEngine()
                            st.session_state.audio_bytes = audio.synthesize(st.session_state.generated_poem)
                            audio_status.update(label="Audio Ready", state="complete")
                    
                        if st.session_state.audio_bytes:
                            audio_placeholder.audio(st.session_state.audio_bytes, format="audio/mpeg")
            else:
                st.info("Waiting for Vision Analysis to retrieve memories...")
import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
import time

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key (Ensure this is kept secure)
GROQ_API_KEY = "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"
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Load whisper model at startup
@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("small")
    except ImportError:
        st.error("❌ Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading Whisper model: {str(e)}")
        st.stop()

# Initialize RAG system internally
def initialize_rag_system():
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        return None

# Load models on startup
whisper_model = load_whisper_model()
rag_llm = initialize_rag_system()

def process_transaction_message(message, llm):
    if llm is None:
        return "Error: RAG system is not initialized."
    system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like amount, merchant, date, mode of payment, and recipient. "
        "If mode of payment is not mentioned, assume cash by default. "
        "If any field is missing, set it as null. "
        "As input is processed through a STT model, it may have mistakes, so validate carefully. "
        "Return only a JSON or a list of JSON objects."
    )
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response.content if hasattr(response, 'content') else response

def main():
    st.markdown("<h1 style='text-align: center;'>🔊 Audio Transaction Processor</h1>", unsafe_allow_html=True)
    st.success("✅ Whisper model and RAG system loaded and ready!")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
        st.markdown("**🎵 Audio Preview:**")
        st.audio(uploaded_file, format="audio/*")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        if st.button('🎤 Transcribe Audio'):
            try:
                with st.spinner("⏳ Transcribing audio... Please wait."):
                    result = whisper_model.transcribe(tmp_file_path)
                    transcription = result.get("text", "")
                
                if not transcription:
                    st.error("❌ No transcription output. Please check your audio file.")
                    return
                
                # Store transcription in session state
                st.session_state.transcription = transcription
                
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # If transcription exists, show editable text area
    if 'transcription' in st.session_state:
        st.markdown("### ✏️ Edit Transcription Before Processing")
        edited_transcription = st.text_area("", st.session_state.transcription, height=200)
        
        if st.button('🚀 Process Transaction Details'):
            with st.spinner("🤖 Processing transaction details..."):
                processed_result = process_transaction_message(edited_transcription, rag_llm)
                if processed_result:
                    st.markdown("### 💰 Extracted Transaction Details")
                    st.code(processed_result, language="json")
                else:
                    st.error("❌ Failed to process transaction details.")

if __name__ == "__main__":
    main()

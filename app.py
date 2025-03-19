import streamlit as st
import tempfile
import os
import json
from langchain_groq import ChatGroq

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key and model settings
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_GROQ_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024

# Load whisper model at startup
@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("small")
    except ImportError:
        st.error("Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        st.stop()

# Initialize RAG system internally
@st.cache_resource
def initialize_rag_system():
    try:
        return ChatGroq(
            api_key=DEFAULT_GROQ_API_KEY,
            model=DEFAULT_GROQ_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Process transaction messages
def process_transaction_message(message, llm):
    system_prompt = (
        "Your input is a voice-transcribed text regarding a transaction. Extract structured details: "
        "\nExample: 'today I spent 500 at Domino's' should be understood as spending at Domino's." 
        "Handle potential transcription errors (e.g., 'tomato' instead of 'Zomato')." 
        "\nIf details are missing, return null values in the JSON."
    )
    
    input_prompt = f"{system_prompt}\nMessage: {message}"
    response = llm.invoke(input_prompt)
    return response

def main():
    st.title("Audio Transaction Processor with Whisper & Groq LLM")
    
    # Load Whisper model
    with st.spinner("Loading Whisper model... This may take a moment."):
        whisper_model = load_whisper_model()
    st.success("Whisper model loaded and ready!")
    
    # Initialize RAG system internally
    llm = initialize_rag_system()
    if not llm:
        st.error("Failed to initialize the RAG system. Please check the API key or model settings.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format="audio/*")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Transcribe button
        if st.button('Transcribe and Process'):
            try:
                with st.spinner("Transcribing audio..."):
                    result = whisper_model.transcribe(tmp_file_path)
                    transcription = result["text"]
                
                # Display transcription
                st.subheader("Transcription Result")
                st.text_area("", transcription, height=200)
                
                # Process transcription with Groq LLM
                with st.spinner("Processing transaction details..."):
                    processed_result = process_transaction_message(transcription, llm)
                    transaction_data = json.loads(processed_result.content)
                    
                    st.subheader("Extracted Transaction Details (Editable):")
                    edited_transaction_data = st.text_area("Edit JSON Data", json.dumps(transaction_data, indent=4), height=300)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Submit"):
                            st.success("Transaction data submitted successfully!")
                    with col2:
                        if st.button("Cancel"):
                            st.warning("Changes discarded.")
                
                # Clean up temp file
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

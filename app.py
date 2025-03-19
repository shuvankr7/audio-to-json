import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key
DEFAULT_GROQ_API_KEY = "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"

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

def initialize_rag_system(groq_api_key, groq_model, temperature, max_tokens):
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def process_transaction_message(message, llm):
    system_prompt = (
        "your gave some input regarding their transaction in voice that is transferred in text and text is your input. "
        "You need to find these details from the text:"
        "as human giving input so input can be of few world and less structured gramatically and simple"
        "example 1: today I spent 500 at dominoze"
        "As input is processed through a STT model so input can have mistakes too like - I spent 500 at tomato, where it is zomato not tomato,you need to think and validate"
        "\n{\"Amount\":105,\n\"Transaction Type\":\"Debit\",\n\"Bank Name\":\"SBI\",\n\"Card Type\":\"Credit Card\",\n\"marchent\":\"Auto Fuel Station\",\n\"paied to whom\":\"Auto Fuel Station\",\n\"Transaction Mode\":\"Credit Card\",\n\"Transaction Date\":\"19-03-25\",\n\"Reference Number\":\"507775912830\",\n\"tag\":[\"Transport\"]\n}"
        "If all the details are not in the input, then the following values of the JSON should be null."
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
    
    # API Key Input
    api_key = st.text_input("GROQ API Key", value=DEFAULT_GROQ_API_KEY, type="password")
    
    # Model Selection
    model = st.selectbox(
        "Select GROQ Model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Temperature and Max Tokens
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with col2:
        max_tokens = st.number_input("Max Tokens", min_value=10, max_value=4096, value=1024, step=10)
    
    # Initialize RAG system
    if st.button("Initialize RAG System"):
        llm = initialize_rag_system(api_key, model, temperature, max_tokens)
        if llm:
            st.session_state.llm = llm
            st.success("RAG system initialized successfully!")
    
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
                if 'llm' not in st.session_state:
                    st.warning("Please initialize the RAG system first.")
                else:
                    with st.spinner("Processing transaction details..."):
                        processed_result = process_transaction_message(transcription, st.session_state.llm)
                        st.subheader("Extracted Transaction Details:")
                        st.code(processed_result.content, language="json")
                
                # Clean up temp file
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

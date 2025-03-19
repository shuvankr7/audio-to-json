import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq

# Set environment variables before imports
os.environ["USER_AGENT"] = "RAG-Chat-Assistant/1.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default Groq API key
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
        st.error("Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
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
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Load Whisper model and RAG system on startup
whisper_model = load_whisper_model()
rag_llm = initialize_rag_system()

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
    st.success("Whisper model and RAG system loaded and ready!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
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
                
                st.subheader("Transcription Result")
                st.text_area("", transcription, height=200)
                
                # Process transcription with Groq LLM
                with st.spinner("Processing transaction details..."):
                    processed_result = process_transaction_message(transcription, rag_llm)
                    st.subheader("Extracted Transaction Details:")
                    st.code(processed_result.content, language="json")
                
                os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

import streamlit as st
import whisper
import tempfile
import os
from moviepy.editor import VideoFileClip
import torch
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="Video to Text Converter",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Video to Text Converter for Indic Languages")
st.markdown("Convert your videos to text in multiple Indian languages using AI")

# Sidebar for language selection
st.sidebar.header("Settings")

# Supported Indic languages in Whisper
INDIC_LANGUAGES = {
    "Hindi": "hi",
    "Bengali": "bn", 
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Assamese": "as",
    "Nepali": "ne"
}

selected_language = st.sidebar.selectbox(
    "Select Language",
    options=list(INDIC_LANGUAGES.keys()),
    index=0
)

# Model selection
model_size = st.sidebar.selectbox(
    "Select Model Size",
    options=["tiny", "base", "small", "medium", "large"],
    index=2,
    help="Larger models are more accurate but slower"
)

@st.cache_resource
def load_whisper_model(model_name):
    """Load and cache the Whisper model"""
    return whisper.load_model(model_name)

def extract_audio_from_video(video_file):
    """Extract audio from video file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Load video and extract audio
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()
        audio.close()
        return temp_audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_path, model, language_code):
    """Transcribe audio using Whisper"""
    try:
        result = model.transcribe(
            audio_path, 
            language=language_code,
            fp16=False,
            verbose=False
        )
        return result
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Main interface
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
    help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM"
)

if uploaded_file is not None:
    # Display video info
    st.video(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size / (1024*1024):.2f} MB")
    
    with col2:
        st.write(f"**Selected language:** {selected_language}")
        st.write(f"**Model size:** {model_size}")
    
    if st.button("🚀 Start Transcription", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded file
            status_text.text("📁 Saving video file...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name
            
            # Step 2: Extract audio
            status_text.text("🎵 Extracting audio from video...")
            progress_bar.progress(30)
            
            audio_path = extract_audio_from_video(temp_video_path)
            
            if audio_path:
                # Step 3: Load model
                status_text.text("🤖 Loading AI model...")
                progress_bar.progress(50)
                
                model = load_whisper_model(model_size)
                
                # Step 4: Transcribe
                status_text.text("🎯 Transcribing audio...")
                progress_bar.progress(70)
                
                language_code = INDIC_LANGUAGES[selected_language]
                result = transcribe_audio(audio_path, model, language_code)
                
                if result:
                    progress_bar.progress(100)
                    status_text.text("✅ Transcription completed!")
                    
                    # Display results
                    st.success("Transcription completed successfully!")
                    
                    # Full transcription
                    st.subheader("📝 Full Transcription")
                    st.text_area(
                        "Transcribed Text",
                        value=result['text'],
                        height=300,
                        help="You can copy this text"
                    )
                    
                    # Segment-wise transcription
                    if 'segments' in result:
                        st.subheader("⏰ Timestamped Segments")
                        
                        segments_data = []
                        for segment in result['segments']:
                            start_time = f"{int(segment['start'] // 60):02d}:{int(segment['start'] % 60):02d}"
                            end_time = f"{int(segment['end'] // 60):02d}:{int(segment['end'] % 60):02d}"
                            segments_data.append({
                                "Time": f"{start_time} - {end_time}",
                                "Text": segment['text'].strip()
                            })
                        
                        st.dataframe(segments_data, use_container_width=True)
                    
                    # Download options
                    st.subheader("💾 Download Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download as text file
                        st.download_button(
                            label="📄 Download as TXT",
                            data=result['text'],
                            file_name=f"transcription_{selected_language.lower()}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Download as SRT (subtitle file)
                        if 'segments' in result:
                            srt_content = ""
                            for i, segment in enumerate(result['segments'], 1):
                                start_time = format_time_srt(segment['start'])
                                end_time = format_time_srt(segment['end'])
                                srt_content += f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"
                            
                            st.download_button(
                                label="🎬 Download as SRT",
                                data=srt_content,
                                file_name=f"subtitles_{selected_language.lower()}.srt",
                                mime="text/plain"
                            )
                
                # Cleanup temporary files
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_bar.empty()
            status_text.empty()

def format_time_srt(seconds):
    """Format time for SRT subtitle format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and OpenAI Whisper</p>
        <p><em>Supports 12+ Indian languages with high accuracy</em></p>
    </div>
    """,
    unsafe_allow_html=True
)

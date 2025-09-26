import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO
import re
from typing import List, Tuple
import zipfile
import google.generativeai as genai
import json
import base64
import time

# Set page config
st.set_page_config(
    page_title="SRT Media Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .config-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    
    .info-box {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        padding: 1rem;
        border-radius: 5px;
        color: #0d47a1;
    }
    
    .video-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API clients
@st.cache_resource
def init_apis():
    try:
        replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        minimax_api_key = st.secrets["MINIMAX_API_KEY"]
        return replicate_client, gemini_model, minimax_api_key
    except Exception as e:
        st.error("❌ Please configure your API keys in Streamlit secrets")
        st.stop()

def parse_srt(srt_content: str) -> List[Tuple[str, str, str, str]]:
    """Parse SRT content and return list of (timestamp, start_time, end_time, text) tuples"""
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    subtitles = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            seq_num = lines[0].strip()
            timestamp = lines[1].strip()
            text = ' '.join(lines[2:]).strip()
            
            if ' --> ' in timestamp:
                start_time, end_time = timestamp.split(' --> ')
                subtitles.append((timestamp.strip(), start_time.strip(), end_time.strip(), text))
    
    return subtitles

def group_subtitles(subtitles: List[Tuple[str, str, str, str]], group_size: int = 2) -> List[Tuple[str, str]]:
    """Group subtitles by specified size"""
    grouped_entries = []
    
    for i in range(0, len(subtitles), group_size):
        group = subtitles[i:i+group_size]
        timestamp = group[0][1]
        clean_timestamp = re.sub(r'[^\w:,\-_]', '_', timestamp)
        
        combined_text = " ".join([subtitle[3].strip() for subtitle in group if subtitle[3].strip()])
        
        if combined_text:
            grouped_entries.append((clean_timestamp, combined_text))
    
    return grouped_entries

def process_individual_subtitles(subtitles: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """Process each subtitle individually"""
    entries = []
    for timestamp, start_time, end_time, text in subtitles:
        text = text.strip()
        if text:
            clean_timestamp = re.sub(r'[^\w:,\-_]', '_', start_time)
            entries.append((clean_timestamp, text))
    return entries

def describe_scene_with_gemini(text: str, style_prompt: str, model) -> str:
    """Use Gemini to describe scenes for image generation"""
    prompt = f"""Create a visual scene description optimized for AI image generation from this subtitle:

SUBTITLE: "{text}"

Create a 25-40 word description focusing on:
- Visual actions and settings
- Character positioning and expressions  
- Lighting and atmosphere
- Cinematic composition

Style requirements: {style_prompt if style_prompt else 'cinematic realism'}

Return only the optimized scene description."""

    try:
        response = model.generate_content(prompt)
        description = response.text.strip().strip('"\'')
        
        if style_prompt and style_prompt.lower() not in description.lower():
            description = f"{description}, {style_prompt}"
        
        return description
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return create_fallback_description(text, style_prompt)
        else:
            st.warning(f"Gemini API error: {str(e)}. Using fallback.")
            return create_fallback_description(text, style_prompt)

def create_fallback_description(text: str, style_prompt: str = "") -> str:
    """Create fallback descriptions when Gemini is unavailable"""
    text_lower = text.lower()
    
    # Detect scene type
    if any(word in text_lower for word in ['running', 'walking', 'moving']):
        base = "Person in dynamic motion"
    elif any(word in text_lower for word in ['talking', 'speaking', 'conversation']):
        base = "Characters in conversation"
    elif any(word in text_lower for word in ['looking', 'watching', 'staring']):
        base = "Focused observation scene"
    else:
        base = "Cinematic scene"
    
    # Detect setting
    if any(word in text_lower for word in ['house', 'home', 'room']):
        setting = "indoor domestic environment"
    elif any(word in text_lower for word in ['office', 'work']):
        setting = "professional office setting"
    elif any(word in text_lower for word in ['outside', 'street', 'car']):
        setting = "urban outdoor location"
    else:
        setting = "atmospheric setting"
    
    # Include the subtitle text in the description for context
    description = f"{base} in {setting}, depicting: {text}, cinematic lighting, professional composition"
    
    if style_prompt:
        description += f", {style_prompt}"
    
    return description

def generate_image(prompt: str, client) -> Image.Image:
    """Generate image using Flux model"""
    try:
        output = client.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "width": 1024,
                "height": 574,
                "num_outputs": 1,
                "num_inference_steps": 4
            }
        )
        
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# MINIMAX AUDIO GENERATION FUNCTIONS (Proper API Workflow)
def create_speech_task(text: str, voice_id: str, speed: float, vol: float, 
                      pitch: float, english_normalization: bool, api_key: str,
                      sample_rate: int = 32000, bitrate: int = 128000, 
                      file_format: str = "wav", model: str = "speech-2.5-hd-preview"):
    """Step 1: Create a speech generation task and get task_id"""
    
    url = "https://api.minimax.io/v1/t2a_async_v2"
    
    headers = {
        "Authorization": f"Bearer {minimax_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "text": text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": vol,
            "pitch": pitch,
            "english_normalization": english_normalization
        },
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": file_format,
            "channel": 1
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            task_data = response.json()
            
            # Check for MiniMax error format
            if 'base_resp' in task_data:
                base_resp = task_data['base_resp']
                status_code = base_resp.get('status_code', 0)
                status_msg = base_resp.get('status_msg', 'Unknown error')
                
                if status_code != 0:
                    return None, f"MiniMax Error {status_code}: {status_msg}"
            
            # Extract task_id
            if 'task_id' in task_data:
                return task_data['task_id'], None
            elif 'data' in task_data and 'task_id' in task_data['data']:
                return task_data['data']['task_id'], None
            else:
                return None, f"No task_id in response: {task_data}"
                
        else:
            return None, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return None, f"Request failed: {str(e)}"

def check_task_status(task_id: str, api_key: str):
    """Step 2: Check the status of a speech generation task"""
    
    url = f"https://api.minimax.chat/v1/query/t2a_v2/{task_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            status_data = response.json()
            
            # Check for MiniMax error format
            if 'base_resp' in status_data:
                base_resp = status_data['base_resp']
                status_code = base_resp.get('status_code', 0)
                status_msg = base_resp.get('status_msg', 'Unknown error')
                
                if status_code != 0:
                    return "failed", None, f"Error {status_code}: {status_msg}"
            
            # Extract status and file_id
            if 'data' in status_data:
                data = status_data['data']
                task_status = data.get('status', 'unknown')
                file_id = data.get('file_id', None)
                return task_status, file_id, None
            else:
                task_status = status_data.get('status', 'unknown')
                file_id = status_data.get('file_id', None)
                return task_status, file_id, None
                
        else:
            return "failed", None, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return "failed", None, f"Request failed: {str(e)}"

def download_audio_file(file_id: str, api_key: str):
    """Step 3: Download the audio file using file_id"""
    
    url = f"https://api.minimax.chat/v1/files/retrieve"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "file_id": file_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            # Check if response contains download URL or direct audio data
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                file_data = response.json()
                
                # Check for error in response
                if 'base_resp' in file_data:
                    base_resp = file_data['base_resp']
                    status_code = base_resp.get('status_code', 0)
                    if status_code != 0:
                        return None, f"Download error {status_code}: {base_resp.get('status_msg', 'Unknown error')}"
                
                # Look for download URL
                if 'file' in file_data and 'download_url' in file_data['file']:
                    download_url = file_data['file']['download_url']
                    
                    # Download from URL
                    audio_response = requests.get(download_url, timeout=60)
                    if audio_response.status_code == 200:
                        return audio_response.content, None
                    else:
                        return None, f"Failed to download from URL: HTTP {audio_response.status_code}"
                
                elif 'download_url' in file_data:
                    download_url = file_data['download_url']
                    
                    # Download from URL
                    audio_response = requests.get(download_url, timeout=60)
                    if audio_response.status_code == 200:
                        return audio_response.content, None
                    else:
                        return None, f"Failed to download from URL: HTTP {audio_response.status_code}"
                
                else:
                    return None, f"No download URL in response: {file_data}"
            
            elif 'audio' in content_type:
                # Direct audio content
                return response.content, None
            
            else:
                return None, f"Unexpected content type: {content_type}"
                
        else:
            return None, f"Download failed: HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return None, f"Download request failed: {str(e)}"

def generate_audio_minimax(text: str, voice_id: str, speed: float, vol: float, 
                          pitch: float, english_normalization: bool, api_key: str,
                          sample_rate: int = 32000, bitrate: int = 128000, 
                          file_format: str = "wav", model: str = "speech-2.5-hd-preview"):
    """Complete MiniMax TTS workflow: Create task -> Poll status -> Download audio"""
    
    # Step 1: Create speech generation task
    st.info("📝 Step 1: Creating speech generation task...")
    task_id, error = create_speech_task(
        text, voice_id, speed, vol, pitch, english_normalization, 
        api_key, sample_rate, bitrate, file_format, model
    )
    
    if error:
        st.error(f"❌ Task creation failed: {error}")
        
        # Provide specific troubleshooting for common errors
        if "1000" in error:
            st.error("🔧 **Error 1000 troubleshooting:**")
            st.error("• Try voice_male_1 or voice_female_1")
            st.error("• Use a different model (speech-01-turbo)")
            st.error("• Shorten your text")
            st.error("• Check if your account has access to this model")
        elif "2049" in error:
            st.error("🔧 **Invalid API key - please check your MINIMAX_API_KEY**")
        elif "2050" in error:
            st.error("🔧 **Insufficient credits - check your account balance**")
        elif "2051" in error:
            st.error("🔧 **Model access denied - try a different model**")
        
        return None
    
    st.success(f"✅ Task created successfully! Task ID: {task_id}")
    
    # Step 2: Poll task status
    st.info("⏳ Step 2: Checking task status...")
    
    max_attempts = 60  # 60 attempts with 2-second intervals = 2 minutes max
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for attempt in range(max_attempts):
        status_text.text(f"⏳ Checking status... (attempt {attempt + 1}/{max_attempts})")
        progress = min(0.9, attempt / max_attempts)
        progress_bar.progress(progress)
        
        task_status, file_id, error = check_task_status(task_id, api_key)
        
        if error:
            st.error(f"❌ Status check failed: {error}")
            return None
        
        if task_status == "Success" or task_status == "completed":
            progress_bar.progress(1.0)
            status_text.success("✅ Task completed successfully!")
            
            if not file_id:
                st.error("❌ Task completed but no file_id received")
                return None
            
            # Step 3: Download audio file
            st.info(f"📥 Step 3: Downloading audio file (ID: {file_id})...")
            st.warning("⚠️ Note: Download URL expires in 9 hours")
            
            audio_data, download_error = download_audio_file(file_id, api_key)
            
            if download_error:
                st.error(f"❌ Download failed: {download_error}")
                return None
            
            st.success(f"🎉 Audio downloaded successfully! ({len(audio_data)} bytes)")
            return audio_data
            
        elif task_status == "Failed" or task_status == "failed":
            progress_bar.progress(1.0)
            status_text.error("❌ Task failed")
            return None
            
        elif task_status in ["Processing", "processing", "Pending", "pending", "Running", "running"]:
            # Task still in progress, continue polling
            time.sleep(2)
            continue
            
        else:
            st.warning(f"⚠️ Unknown status: {task_status}")
            time.sleep(2)
            continue
    
    # Timeout
    progress_bar.progress(1.0)
    status_text.error("❌ Task timeout - processing took too long")
    st.error("🔧 Try with shorter text or try again later")
    return None

def generate_video(audio_file, images_zip, prompt: str, max_attempts: int, client):
    """Generate video using FFmpeg model"""
    try:
        output = client.run(
            "fofr/smart-ffmpeg",
            input={
                "files": [audio_file, images_zip],
                "prompt": prompt,
                "max_attempts": max_attempts
            }
        )
        
        return output
    except Exception as e:
        st.error(f"Error generating video: {str(e)}")
        return None

# Initialize APIs
replicate_client, gemini_model, minimax_api_key = init_apis()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 SRT Media Generator</h1>
    <p>Transform subtitle files into stunning visuals and videos</p>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["🎤 Audio Generation", "🖼️ Image Generation", "🎥 Video Creation"])

# ===== AUDIO GENERATION TAB =====
with tab1:
    # Sidebar Configuration for Audio Generation
    with st.sidebar:
        st.header("🎤 Audio Configuration")
        
        voice_id = st.selectbox(
            "Voice Selection",
            options=[
                "voice_male_1",
                "voice_female_1", 
                "voice_male_2",
                "voice_female_2",
                "voice_male_3",
                "voice_female_3",
                "English_expressive_narrator",
                "English_cheerful_male",
                "English_calm_female",
                "English_professional_male",
                "English_warm_female"
            ],
            index=0,
            help="Choose the voice character for audio generation"
        )
        
        st.markdown("---")
        
        speed = st.slider(
            "🚀 Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Audio playback speed"
        )
        
        vol = st.slider(
            "🔊 Volume",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Audio volume level"
        )
        
        pitch = st.slider(
            "🎵 Pitch",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Audio pitch adjustment"
        )
        
        st.markdown("---")
        
        # Audio quality settings
        st.subheader("🎧 Audio Quality")
        
        sample_rate = st.selectbox(
            "Sample Rate",
            options=[16000, 22050, 32000, 44100, 48000],
            index=2,  # 32000 default
            help="Audio sample rate in Hz"
        )
        
        bitrate = st.selectbox(
            "Bitrate",
            options=[64000, 96000, 128000, 192000, 256000, 320000],
            index=2,  # 128000 default
            help="Audio bitrate in bps"
        )
        
        file_format = st.selectbox(
            "Format",
            options=["wav", "mp3"],
            index=0,
            help="Output audio format"
        )
        
        english_normalization = st.checkbox(
            "English Normalization",
            value=False,
            help="Enable English text normalization"
        )
        
        st.markdown("---")
        
        # Alternative models section
        st.subheader("🔄 Alternative Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            alternative_model = st.selectbox(
                "Try Different Model",
                options=[
                    "speech-2.5-hd-preview",
                    "speech-01-turbo", 
                    "speech-01",
                    "tts-1",
                    "tts-1-hd"
                ],
                index=0,
                help="Try different models if one doesn't work"
            )
        
        with col2:
            use_alternative = st.checkbox(
                "Use Alternative Model",
                help="Enable to use the selected alternative model"
            )
        
        st.info("💡 If you get Error 1000, try:\n• Different voice (voice_male_1, voice_female_1)\n• Alternative model\n• Shorter text\n• Wait and retry")

    # Main Content for Audio Generation
    st.subheader("🎤 Text-to-Speech Generator")
    
    st.markdown("""
    <div class="info-box">
        🎯 <strong>Audio Generation Process:</strong><br>
        1. Enter your text content<br>
        2. Choose voice and adjust settings<br>
        3. Generate high-quality audio with MiniMax<br>
        4. Download your audio file!
    </div>
    """, unsafe_allow_html=True)
    
    # Text input for audio generation
    audio_text = st.text_area(
        "📝 Text to Convert",
        placeholder="Enter the text you want to convert to speech...",
        height=150,
        help="Enter the text that will be converted to audio using MiniMax AI voice synthesis"
    )
    
    # SRT file option for audio generation
    st.markdown("---")
    st.subheader("📁 Or Upload SRT File for Audio")
    
    srt_for_audio = st.file_uploader(
        "Choose SRT file to convert to audio",
        type=['srt'],
        help="Upload an SRT file to extract and convert text to audio",
        key="audio_srt_upload"
    )
    
    # Process SRT for audio if uploaded
    if srt_for_audio is not None:
        try:
            try:
                srt_content = srt_for_audio.read().decode('utf-8')
            except UnicodeDecodeError:
                srt_for_audio.seek(0)
                srt_content = srt_for_audio.read().decode('latin-1')
            
            subtitles = parse_srt(srt_content)
            
            if subtitles:
                # Extract all text from subtitles
                all_text = " ".join([subtitle[3] for subtitle in subtitles])
                
                st.success(f"✅ Extracted text from {len(subtitles)} subtitles")
                
                # Option to use extracted text
                if st.button("📋 Use Extracted Text", width="stretch", key="use_extracted_text"):
                    audio_text = all_text
                    st.text_area("📝 Extracted Text", value=all_text, height=150, key="extracted_text_display")
                
                # Preview extracted text
                with st.expander("👀 Preview Extracted Text"):
                    st.write(all_text[:500] + "..." if len(all_text) > 500 else all_text)
        
        except Exception as e:
            st.error(f"❌ Error processing SRT file: {str(e)}")
    
    # Generate audio button and cost estimate
    if audio_text.strip():
        # Cost estimate (MiniMax pricing may differ)
        word_count = len(audio_text.split())
        char_count = len(audio_text)
        estimated_audio_cost = char_count * 0.0001  # Rough estimate for MiniMax
        
        st.markdown(f"""
        <div class="info-box">
            📊 <strong>Text Stats:</strong> {word_count} words, {char_count} characters<br>
            💰 <strong>Estimated cost:</strong> ~${estimated_audio_cost:.4f} USD<br>
            🎤 <strong>Voice:</strong> {voice_id}<br>
            🎧 <strong>Quality:</strong> {sample_rate}Hz, {bitrate}bps, {file_format.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🎤 Generate Audio")
        
        if st.button("🚀 Generate Audio", type="primary", width="stretch", key="generate_audio_btn"):
            with st.spinner("🎤 Generating audio with MiniMax... This may take a few moments..."):
                try:
                    # Determine which model to use
                    model_to_use = alternative_model if use_alternative else "speech-2.5-hd-preview"
                    
                    audio_data = generate_audio_minimax(
                        text=audio_text,
                        voice_id=voice_id,
                        speed=speed,
                        vol=vol,
                        pitch=pitch,
                        english_normalization=english_normalization,
                        api_key=minimax_api_key,
                        sample_rate=sample_rate,
                        bitrate=bitrate,
                        file_format=file_format,
                        model=model_to_use
                    )
                    
                    if audio_data:
                        st.success("🎉 Audio generated successfully with MiniMax!")
                        
                        # Display audio player
                        st.audio(audio_data, format=f'audio/{file_format}')
                        
                        # Download audio
                        file_extension = "wav" if file_format == "wav" else "mp3"
                        st.download_button(
                            label="💾 Download Audio",
                            data=audio_data,
                            file_name=f"minimax_generated_audio.{file_extension}",
                            mime=f"audio/{file_format}",
                            width="stretch",
                            key="download_audio"
                        )
                        
                        # Store audio in session state for video creation
                        st.session_state.generated_audio = audio_data
                        st.session_state.generated_audio_format = file_format
                    else:
                        st.error("❌ Failed to generate audio")
                
                except Exception as e:
                    st.error(f"❌ Error generating audio: {str(e)}")
                    st.info("💡 Please check your MiniMax API key and internet connection")
    
    else:
        st.info("👆 Enter text above or upload an SRT file to generate audio")

# ===== IMAGE GENERATION TAB =====
with tab2:
    # Sidebar Configuration for Image Generation
    with st.sidebar:
        st.header("⚙️ Image Configuration")
        
        processing_mode = st.radio(
            "Subtitle Processing",
            ["Individual subtitles", "Group subtitles"],
            help="Choose how to process subtitle entries"
        )
        
        if processing_mode == "Group subtitles":
            group_size = st.number_input(
                "Group Size",
                min_value=2,
                max_value=50,
                value=2,
                help="Number of subtitles to group together"
            )
        else:
            group_size = 1
        
        description_mode = st.radio(
            "Scene Description",
            ["Enhanced AI descriptions", "Basic descriptions"],
            help="AI descriptions provide better visual prompts"
        )
        
        st.markdown("---")
        
        style_prompt = st.text_area(
            "🎨 Visual Style (Optional)",
            placeholder="e.g., 'cyberpunk neon', 'watercolor painting', 'film noir'",
            height=80,
            help="Add consistent visual style to all images"
        )
        
        st.markdown("---")
        st.info("📐 Images: 1024×574px (16:9)")
        
        force_fallback = st.checkbox(
            "Skip AI descriptions",
            help="Use basic descriptions only"
        )

    # Main Content for Image Generation
    col1, col2 = st.columns([2, 1])

    with col1:
        # File Upload
        st.subheader("📁 Upload SRT File")
        uploaded_file = st.file_uploader(
            "Choose your subtitle file",
            type=['srt'],
            help="Upload a valid .srt subtitle file",
            key="img_srt_upload"
        )

    with col2:
        if uploaded_file:
            st.subheader("📊 Quick Stats")
            # We'll populate this after parsing

    # Process uploaded file
    if uploaded_file is not None:
        # Parse SRT
        try:
            try:
                srt_content = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                srt_content = uploaded_file.read().decode('latin-1')
            
            subtitles = parse_srt(srt_content)
            
            if not subtitles:
                st.error("❌ No subtitles found. Please check your SRT file format.")
                st.stop()
            
            # Update stats
            with col2:
                st.metric("Total Subtitles", len(subtitles))
                
                # Estimate duration
                if len(subtitles) > 0:
                    try:
                        last_time = subtitles[-1][2]  # end_time of last subtitle
                        time_parts = last_time.split(':')
                        total_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                        st.metric("Duration", f"~{total_minutes} min")
                    except:
                        st.metric("Duration", "Unknown")
            
            # Process subtitles based on mode
            if processing_mode == "Individual subtitles":
                processed_entries = process_individual_subtitles(subtitles)
            else:  # Group subtitles
                processed_entries = group_subtitles(subtitles, group_size)
            
            st.markdown(f"""
            <div class="success-box">
                ✅ Successfully processed <strong>{len(processed_entries)}</strong> entries from <strong>{len(subtitles)}</strong> subtitles
            </div>
            """, unsafe_allow_html=True)
            
            # Preview processed entries
            with st.expander("👀 Preview Processed Entries"):
                show_all_entries = st.checkbox("Show all entries", key="show_all_processed")
                
                entries_to_show = processed_entries if show_all_entries else processed_entries[:5]
                
                for i, (timestamp, text) in enumerate(entries_to_show):
                    st.markdown(f"**{i+1}.** `{timestamp}`")
                    st.write(text)
                    st.markdown("---")
                
                if not show_all_entries and len(processed_entries) > 5:
                    st.info(f"Showing first 5 of {len(processed_entries)} entries. Check 'Show all entries' to see more.")
            
            # Generate scene descriptions
            st.subheader("🎭 Scene Descriptions")
            
            if description_mode == "Enhanced AI descriptions" and not force_fallback:
                st.info("🤖 Creating AI-optimized scene descriptions...")
                
                scene_descriptions = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (timestamp, text) in enumerate(processed_entries):
                    status_text.text(f"Processing {i+1}/{len(processed_entries)}: {text[:50]}...")
                    progress_bar.progress(i / len(processed_entries))
                    
                    description = describe_scene_with_gemini(text, style_prompt, gemini_model)
                    scene_descriptions.append((timestamp, text, description))
                
                progress_bar.progress(1.0)
                status_text.success("✅ All descriptions created!")
            
            else:
                st.info("📝 Using basic scene descriptions...")
                scene_descriptions = []
                for timestamp, text in processed_entries:
                    if style_prompt.strip():
                        description = f"A cinematic scene depicting: {style_prompt.strip()}, {text}"
                    else:
                        description = f"A cinematic scene depicting: {text}"
                    scene_descriptions.append((timestamp, text, description))
            
            # Preview scene descriptions
            with st.expander("🎬 Preview Scene Descriptions"):
                show_all_descriptions = st.checkbox("Show all descriptions", key="show_all_descriptions")
                
                descriptions_to_show = scene_descriptions if show_all_descriptions else scene_descriptions[:3]
                
                for i, (timestamp, original, description) in enumerate(descriptions_to_show):
                    st.markdown(f"**{i+1}.** `{timestamp}`")
                    st.markdown(f"*Original:* {original}")
                    st.markdown(f"*Description:* {description}")
                    st.markdown("---")
                
                if not show_all_descriptions and len(scene_descriptions) > 3:
                    st.info(f"Showing first 3 of {len(scene_descriptions)} descriptions. Check 'Show all descriptions' to see more.")
            
            # Cost estimate
            estimated_cost = len(scene_descriptions) * 0.003
            st.markdown(f"""
            <div class="info-box">
                💰 <strong>Estimated cost:</strong> ~${estimated_cost:.3f} USD for {len(scene_descriptions)} images
            </div>
            """, unsafe_allow_html=True)
            
            # Generate images
            st.subheader("🎨 Generate Images")
            
            col_gen1, col_gen2 = st.columns(2)
            
            with col_gen1:
                generate_all = st.button("🚀 Generate All Images", type="primary", width="stretch", key="gen_all_img")
            
            with col_gen2:
                if len(scene_descriptions) > 5:
                    generate_sample = st.button("🎯 Generate 5 Sample Images", width="stretch", key="gen_sample_img")
                else:
                    generate_sample = False
            
            if generate_all or generate_sample:
                descriptions_to_use = scene_descriptions[:5] if generate_sample else scene_descriptions
                
                # Initialize session state for storing images if not exists
                if 'generated_images' not in st.session_state:
                    st.session_state.generated_images = []
                if 'image_data_for_download' not in st.session_state:
                    st.session_state.image_data_for_download = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                generated_images = []
                image_data_for_download = []
                
                for i, (timestamp, original_text, description) in enumerate(descriptions_to_use):
                    status_text.text(f"Generating image {i+1}/{len(descriptions_to_use)}...")
                    progress_bar.progress(i / len(descriptions_to_use))
                    
                    image = generate_image(description, replicate_client)
                    
                    if image:
                        generated_images.append((image, description, original_text, timestamp))
                        
                        # Prepare for download with sequential naming
                        buf = BytesIO()
                        image.save(buf, format='PNG')
                        buf.seek(0)
                        image_data_for_download.append((buf.getvalue(), f"image{i+1:04d}.png"))
                
                progress_bar.progress(1.0)
                status_text.success(f"🎉 Generated {len(generated_images)} images!")
                
                # Store in session state to prevent disappearing on download
                st.session_state.generated_images = generated_images
                st.session_state.image_data_for_download = image_data_for_download
                
                # Display results
                if st.session_state.generated_images:
                    st.subheader("🖼️ Generated Images")
                    
                    # Bulk download button
                    if len(st.session_state.image_data_for_download) > 1:
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for img_data, filename in st.session_state.image_data_for_download:
                                zip_file.writestr(filename, img_data)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="📦 Download All Images (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="srt_generated_images.zip",
                            mime="application/zip",
                            width="stretch",
                            key="download_all_zip"
                        )
                        st.markdown("---")
                    
                    # Display images
                    for i, (image, description, original_text, timestamp) in enumerate(st.session_state.generated_images):
                        with st.container():
                            st.markdown(f"### 🎬 Scene {i+1}: `image{i+1:04d}.png`")
                            
                            img_col, info_col = st.columns([3, 2])
                            
                            with img_col:
                                st.image(image, width="stretch")
                            
                            with info_col:
                                st.markdown("**📝 Original Subtitle:**")
                                st.write(original_text)
                                
                                st.markdown("**🎭 Scene Description:**")
                                st.write(description)
                                
                                # Individual download
                                buf = BytesIO()
                                image.save(buf, format='PNG')
                                buf.seek(0)
                                st.download_button(
                                    label="💾 Download",
                                    data=buf.getvalue(),
                                    file_name=f"image{i+1:04d}.png",
                                    mime="image/png",
                                    key=f"download_individual_{i}",
                                    width="stretch"
                                )
                            
                            st.markdown("---")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    # Display previously generated images if they exist in session state
    elif 'generated_images' in st.session_state and st.session_state.generated_images:
        st.subheader("🖼️ Previously Generated Images")
        
        # Bulk download button for previous images
        if len(st.session_state.image_data_for_download) > 1:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img_data, filename in st.session_state.image_data_for_download:
                    zip_file.writestr(filename, img_data)
            
            zip_buffer.seek(0)
            st.download_button(
                label="📦 Download All Previous Images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="srt_generated_images.zip",
                mime="application/zip",
                width="stretch",
                key="download_previous_all_zip"
            )
            st.markdown("---")
        
        # Display previous images
        for i, (image, description, original_text, timestamp) in enumerate(st.session_state.generated_images):
            with st.expander(f"🎬 Scene {i+1}: image{i+1:04d}.png"):
                img_col, info_col = st.columns([2, 1])
                with img_col:
                    st.image(image, width="stretch")
                with info_col:
                    st.markdown("**📝 Original:**")
                    st.write(original_text)
                    st.markdown("**🎭 Description:**")
                    st.write(description)
                    
                    # Individual download for previous images
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button(
                        label="💾 Download",
                        data=buf.getvalue(),
                        file_name=f"image{i+1:04d}.png",
                        mime="image/png",
                        key=f"download_previous_{i}",
                        width="stretch"
                    )

# ===== VIDEO CREATION TAB =====
with tab3:
    st.subheader("🎥 Video Creation with FFmpeg")
    
    # Sidebar for Video Configuration
    with st.sidebar:
        st.header("🎬 Video Configuration")
        
        max_attempts = st.number_input(
            "Max Attempts",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum attempts for video processing"
        )
        
        st.markdown("---")
        st.info("📋 Requirements:\n- Audio file\n- Images ZIP file\n- FFmpeg prompt")

    # Video Creation Interface
    st.markdown("""
    <div class="video-box">
        🎯 <strong>Video Creation Process:</strong><br>
        1. Upload your audio file (MP3, WAV, etc.)<br>
        2. Upload a ZIP file containing your images<br>
        3. Provide FFmpeg instructions<br>
        4. Generate your video!
    </div>
    """, unsafe_allow_html=True)
    
    # File uploads for video
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎵 Upload Audio File")
        audio_file = st.file_uploader(
            "Choose audio file",
            type=['mp3', 'wav', 'aac', 'm4a', 'ogg'],
            help="Upload the audio track for your video",
            key="audio_upload"
        )
        
        # Option to use generated audio
        if 'generated_audio' in st.session_state:
            if st.button("🎤 Use Generated Audio", width="stretch", key="use_generated_audio"):
                # Create a temporary file-like object from the generated audio
                audio_buffer = BytesIO(st.session_state.generated_audio)
                audio_file = audio_buffer
                st.success("✅ Using generated MiniMax audio")
                st.audio(st.session_state.generated_audio, format=f'audio/{st.session_state.generated_audio_format}')
        
        if audio_file:
            if not hasattr(audio_file, 'name'):
                st.success("✅ Audio ready: Generated MiniMax audio")
            else:
                st.success(f"✅ Audio uploaded: {audio_file.name}")
                # Show audio player for uploaded files
                st.audio(audio_file)
    
    with col2:
        st.subheader("📦 Upload Images ZIP")
        images_zip = st.file_uploader(
            "Choose images ZIP file",
            type=['zip'],
            help="Upload a ZIP file containing all your images",
            key="images_zip_upload"
        )
        
        if images_zip:
            st.success(f"✅ Images uploaded: {images_zip.name}")
            st.info(f"File size: {len(images_zip.getvalue()) / 1024 / 1024:.2f} MB")
    
    # Quick option to use generated images
    if 'image_data_for_download' in st.session_state and st.session_state.image_data_for_download:
        st.markdown("---")
        st.subheader("🖼️ Use Generated Images")
        
        if st.button("📦 Create ZIP from Generated Images", width="stretch", key="create_zip_from_generated"):
            # Create ZIP from generated images (they're already named sequentially)
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for img_data, filename in st.session_state.image_data_for_download:
                    zip_file.writestr(filename, img_data)
            
            zip_buffer.seek(0)
            st.session_state.generated_images_zip = zip_buffer.getvalue()
            st.success("✅ ZIP file created from generated images!")
            
            # Download option
            st.download_button(
                label="💾 Download Generated Images ZIP",
                data=st.session_state.generated_images_zip,
                file_name="generated_images_for_video.zip",
                mime="application/zip",
                key="download_generated_zip"
            )
    
    # FFmpeg prompt
    st.subheader("⚙️ FFmpeg Instructions")
    ffmpeg_prompt = st.text_area(
        "FFmpeg Prompt",
        placeholder="Example: Create a video by combining the audio file with images, showing each image for 2 seconds, with smooth transitions between images. Add fade in/out effects and ensure audio and video are synchronized.",
        height=120,
        help="Provide detailed instructions for how FFmpeg should process your files"
    )
    
    # Example prompts
    with st.expander("📝 Example FFmpeg Prompts"):
        st.markdown("""
        **Basic Slideshow:**
        ```
        Create a video slideshow using the audio file and images. Show each image for 3 seconds with crossfade transitions.
        ```
        
        **Synchronized Video:**
        ```
        Create a video by syncing images with audio timing. Use subtitle timing to match images with corresponding audio segments.
        ```
        
        **Advanced Effects:**
        ```
        Create a cinematic video with the audio and images. Add ken burns effects, fade transitions, and ensure the final video matches the audio duration perfectly.
        ```
        """)
    
    # Cost estimate for video
    if audio_file and (images_zip or 'generated_images_zip' in st.session_state) and ffmpeg_prompt:
        estimated_video_cost = 0.05  # Rough estimate
        st.markdown(f"""
        <div class="video-box">
            💰 <strong>Estimated cost:</strong> ~${estimated_video_cost:.3f} USD for video generation
        </div>
        """, unsafe_allow_html=True)
    
    # Generate video
    st.subheader("🎬 Generate Video")
    
    if st.button("🚀 Create Video", type="primary", width="stretch", key="generate_video_btn"):
        if not audio_file:
            st.error("❌ Please upload an audio file or use generated audio")
        elif not images_zip and 'generated_images_zip' not in st.session_state:
            st.error("❌ Please upload an images ZIP file or create one from generated images")
        elif not ffmpeg_prompt.strip():
            st.error("❌ Please provide FFmpeg instructions")
        else:
            with st.spinner("🎬 Creating your video... This may take several minutes..."):
                try:
                    # Use generated images ZIP if available, otherwise use uploaded ZIP
                    zip_to_use = st.session_state.get('generated_images_zip') if 'generated_images_zip' in st.session_state else images_zip
                    
                    # Generate video
                    video_result = generate_video(
                        audio_file=audio_file,
                        images_zip=zip_to_use,
                        prompt=ffmpeg_prompt,
                        max_attempts=max_attempts,
                        client=replicate_client
                    )
                    
                    if video_result:
                        st.success("🎉 Video created successfully!")
                        
                        # Display video result
                        if isinstance(video_result, str):
                            # If it's a URL
                            st.video(video_result)
                            
                            # Download video
                            video_response = requests.get(video_result)
                            if video_response.status_code == 200:
                                st.download_button(
                                    label="💾 Download Video",
                                    data=video_response.content,
                                    file_name="generated_video.mp4",
                                    mime="video/mp4",
                                    key="download_video"
                                )
                        elif isinstance(video_result, list) and len(video_result) > 0:
                            # If it's a list of URLs
                            video_url = video_result[0]
                            st.video(video_url)
                            
                            # Download video
                            video_response = requests.get(video_url)
                            if video_response.status_code == 200:
                                st.download_button(
                                    label="💾 Download Video",
                                    data=video_response.content,
                                    file_name="generated_video.mp4",
                                    mime="video/mp4",
                                    key="download_video"
                                )
                        else:
                            st.warning("Video was created but format is unexpected. Check the result:")
                            st.write(video_result)
                    
                except Exception as e:
                    st.error(f"❌ Error creating video: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🚀 Built with Streamlit | 🤖 Powered by Flux AI, Gemini, MiniMax & FFmpeg | Made By Mathew G.</div>", 
    unsafe_allow_html=True
)

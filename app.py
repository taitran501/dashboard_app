import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import os
import re
import time
import glob
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Dashboard Insights - Your Friendly Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# System instruction constant
SYS_INSTRUCTION = """
You are a Senior Business Analyst providing a "First Touch" Executive Summary of a dashboard. 
Your goal is to give the user an immediate understanding of the business health, trends, and outliers based *strictly* on the visual data.

### 1. GUIDING PRINCIPLES
*   **Synthesis over List:** Do not just list every number. Group them logically (e.g., "Revenue is strong, driven by X...").
*   **Visual Logic:** Interpret charts visually. (e.g., "The line chart shows a steady Q1 climb followed by a sharp Q2 drop").
*   **Strict Honesty:** If a number is blurry or missing, state "Not visible/N/A". Do not hallucinate.
*   **Conciseness:** Be direct. Bullet points are preferred.

### 2. REPORT FORMAT (Strict Markdown)

#### 🎯 Executive Snapshot
*   **One-Sentence Verdict:** (e.g., "Performance is trending down due to weak Q3 export volume," or "Strong growth visible in the US market.")
*   **Primary Metrics:** (Extract the biggest/top-most numbers: Total Value, Volume, etc.)

#### 📊 Critical Drivers (Who/What is driving this?)
*   **Top Performers:** (Top 3 Buyers, Products, or Markets with their specific values).
*   **Concentration Risk:** (Insight: Is the revenue dependent on just 1-2 buyers? e.g., "Buyer A accounts for >50% of the total value.")

#### 📈 Trend & Timeline Story
*   **Trajectory:** (Up/Down/Flat/Volatile).
*   **Key Movements:** (e.g., "Peaked in March, then plateaued until June.")
*   **Seasonality:** (Any visible recurring patterns?)

#### 🚨 Anomalies & Red Flags
*   **Outliers:** (Any sudden spikes, zero-value months, or unexpected drops).
*   **Data Gaps:** (Any missing months or unreadable sections).
"""

MODEL_NAME = "gemini-2.5-flash"

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = []
if 'current_key_index' not in st.session_state:
    st.session_state.current_key_index = 0
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = []


def load_api_keys():
    """Load API keys from .env file, key.txt, or environment variables"""
    keys = []
    try:
        # Priority 1: Try loading from .env file (api_key_1 to api_key_8)
        for i in range(1, 9):
            key = os.getenv(f'api_key_{i}')
            if key:
                # Remove quotes if present
                key = key.strip('"\'')
                keys.append(key)
        
        # Priority 2: If no keys from .env, try key.txt file
        if not keys and os.path.exists('key.txt'):
            with open('key.txt', 'r') as f:
                content = f.read()
                # Extract all API keys using regex
                matches = re.findall(r'api_key_\d+="([^"]+)"', content)
                keys = matches
        
        # Priority 3: Fallback to GEMINI_API_KEY_* environment variables
        if not keys:
            for i in range(1, 9):
                key = os.getenv(f'GEMINI_API_KEY_{i}')
                if key:
                    keys.append(key)
    except Exception as e:
        st.error(f"Oops! We had trouble loading your API keys: {e}")
    
    return keys


def get_next_api_key():
    """Get next API key using round-robin rotation"""
    if not st.session_state.api_keys:
        return None
    
    key = st.session_state.api_keys[st.session_state.current_key_index]
    # Move to next key for next request
    st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(st.session_state.api_keys)
    return key


def initialize_model(api_key):
    """Initialize Gemini model with given API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYS_INSTRUCTION
        )
        return model
    except Exception as e:
        return None


def split_image_smart(image):
    """Split image if it's too tall"""
    w, h = image.size
    if h > w * 2.0:
        half_h = h // 2
        return [
            image.crop((0, 0, w, half_h)),
            image.crop((0, half_h, w, h))
        ]
    return [image]


def load_image_from_file(file_input, file_name):
    """Load image from file (PDF, PNG, or JPG) and return PIL Image"""
    try:
        if file_name.lower().endswith('.pdf'):
            # Handle PDF from uploaded file
            file_input.seek(0)  # Reset file pointer
            pdf_bytes = file_input.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) < 1:
                return None, "This PDF file appears to be empty"
            page = doc.load_page(0)
            
            # Zoom logic
            rect = page.rect
            target_width = 1600
            zoom = target_width / rect.width
            if zoom > 2:
                zoom = 2
            if zoom < 0.5:
                zoom = 0.5
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return image, None
        else:
            # Handle image files
            file_input.seek(0)  # Reset file pointer
            image = Image.open(io.BytesIO(file_input.read()))
            return image, None
    except Exception as e:
        return None, f"Sorry, we couldn't read this file: {str(e)}"


def analyze_single_file(file_input, file_name):
    """Analyze a single file (PDF, PNG, or JPG)"""
    # Get API key and initialize model
    api_key = get_next_api_key()
    if not api_key:
        return "Oops! We couldn't access our API keys. Please check your key.txt file for us."
    
    model = initialize_model(api_key)
    if not model:
        return "Sorry, we had trouble connecting to our analysis service. Please try again in a moment."
    
    # 1. Image Loading
    image, error = load_image_from_file(file_input, file_name)
    if error:
        # Make error messages more friendly
        if "Empty PDF" in error:
            return "This PDF file appears to be empty. Could you check the file and try another one?"
        return f"Sorry, we couldn't read this file: {error}"

    # 2. Resize
    w, h = image.size
    max_dimension = 2048
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    images = split_image_smart(image)

    # 3. Prepare content inputs
    content_inputs = []
    user_prompt = """
    Analyze this dashboard image. 
    Provide the 'First Touch' Executive Summary as defined in your system instructions.
    Focus on finding the narrative behind the numbers.
    """

    for img in images:
        content_inputs.append(img)
    content_inputs.append(user_prompt)

    # 4. Inference with retry logic
    max_retries = len(st.session_state.api_keys)  # Try all keys if needed
    for attempt in range(max_retries):
        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8192,
            )

            response = model.generate_content(
                content_inputs,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
                elif candidate.finish_reason == 3:
                    # Safety filter blocked - try next key
                    if attempt < max_retries - 1:
                        api_key = get_next_api_key()
                        model = initialize_model(api_key)
                        continue
                    return "We couldn't analyze this content due to safety filters. Could you try a different file?"
                else:
                    return f"Sorry, we couldn't generate an analysis for this file. Please try again."
            else:
                return "Hmm, we didn't get a response from our analysis service. Could you try again?"

        except Exception as e:
            error_msg = str(e)
            # If rate limit or API error, try next key
            if ("quota" in error_msg.lower() or "rate" in error_msg.lower() or 
                "api" in error_msg.lower()) and attempt < max_retries - 1:
                api_key = get_next_api_key()
                model = initialize_model(api_key)
                time.sleep(1)  # Brief pause before retry
                continue
            return f"Oops, we ran into an issue: {error_msg}. Please try again in a moment."
    
    return "We're having trouble connecting right now. Could you try again in a few moments?"


def main():
    # Inject custom CSS for human-centric design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Apply Nunito font to all elements */
    * html, body, [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, button {
        font-family: 'Nunito', sans-serif !important;
    }
    
    /* Explicitly force Material Icons font for icon elements */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
        font-weight: normal !important; /* Icons shouldn't be bold */
    }
    
    /* Soft color palette */
    :root {
        --primary-color: #6B9BD2;
        --primary-hover: #5A8BC2;
        --background: #FAF9F6;
        --text-color: #4A4A4A;
        --success-bg: #E8F5E9;
        --success-text: #2E7D32;
        --error-bg: #FFEBEE;
        --error-text: #C62828;
        --border-color: #E0E0E0;
    }
    
    /* Main background */
    .stApp {
        background-color: var(--background);
    }
    
    /* Style buttons with rounded corners and soft shadows */
    .stButton > button {
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-hover) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Style text inputs with rounded corners */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(107, 155, 210, 0.1) !important;
    }
    
    /* Style file uploader */
    .stFileUploader {
        border-radius: 12px !important;
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    /* Style success/error messages */
    .stSuccess {
        background-color: var(--success-bg) !important;
        color: var(--success-text) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left: 4px solid var(--success-text) !important;
    }
    
    .stError {
        background-color: var(--error-bg) !important;
        color: var(--error-text) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left: 4px solid var(--error-text) !important;
    }
    
    .stInfo {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left: 4px solid #1565C0 !important;
    }
    
    .stWarning {
        background-color: #FFF3E0 !important;
        color: #E65100 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border-left: 4px solid #E65100 !important;
    }
    
    /* Style headers and text */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        border-radius: 8px !important;
        background-color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Style selectbox */
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }
    
    /* Smooth transitions */
    * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with humanized copy
    st.title("Welcome! Let's analyze your dashboards together")
    st.markdown("Share your dashboard images with us, and we'll help you discover the insights that matter most. We support PDF, PNG, and JPG files.")
    
    # Load API keys
    if not st.session_state.api_keys:
        st.session_state.api_keys = load_api_keys()
        if not st.session_state.api_keys:
            st.error("Oops! We couldn't find your API keys. Please make sure your key.txt file is set up correctly.")
            st.stop()
        else:
            st.success(f"Great! We're ready to help with {len(st.session_state.api_keys)} API keys loaded.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("About This Tool")
        st.info(f"We're using {len(st.session_state.api_keys)} API keys to ensure smooth processing for you.")
        st.markdown("---")
        st.markdown("### What We Support")
        st.markdown("- PDF files")
        st.markdown("- PNG images")
        st.markdown("- JPG/JPEG images")
    
    # File Input Section
    st.header("Share your dashboards with us")
    
    uploaded_files = st.file_uploader(
        "Select your dashboard files",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="You can upload one or more dashboard files at once. We'll analyze each one for you."
    )
    if uploaded_files:
        # Store file data in session state for later preview
        st.session_state.uploaded_files_data = []
        for file in uploaded_files:
            file.seek(0)
            file_data = BytesIO(file.read())
            file_data.name = file.name
            st.session_state.uploaded_files_data.append(file_data)
    
    # Get files to process
    files_to_process = uploaded_files if uploaded_files else st.session_state.uploaded_files_data
    
    # Preview Section for uploaded files
    if files_to_process:
        st.header("Take a look before we analyze")
        
        preview_tab1, preview_tab2 = st.tabs(["📊 Your Dashboard", "📝 Your Results"])
        
        with preview_tab1:
            if files_to_process:
                selected_file_idx = st.selectbox(
                    "Which dashboard would you like to see?",
                    range(len(files_to_process)),
                    format_func=lambda x: files_to_process[x].name if hasattr(files_to_process[x], 'name') else f"File {x+1}"
                )
                
                if selected_file_idx is not None and selected_file_idx < len(files_to_process):
                    selected_file = files_to_process[selected_file_idx]
                    file_name = selected_file.name if hasattr(selected_file, 'name') else f"file_{selected_file_idx+1}"
                    
                    try:
                        # Create a copy of file data to avoid pointer issues
                        selected_file.seek(0)
                        file_copy = BytesIO(selected_file.read())
                        file_copy.name = file_name
                        
                        image, error = load_image_from_file(file_copy, file_name)
                        if error:
                            st.error(error)
                        else:
                            st.image(image, caption=file_name, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading preview: {str(e)}")
        
        with preview_tab2:
            if st.session_state.results:
                selected_result_idx = st.selectbox(
                    "Which result would you like to review?",
                    range(len(st.session_state.results)),
                    format_func=lambda x: st.session_state.results[x]['File Name']
                )
                
                if selected_result_idx is not None and selected_result_idx < len(st.session_state.results):
                    result = st.session_state.results[selected_result_idx]
                    st.markdown(f"**File:** {result['File Name']}")
                    st.markdown(f"**Status:** {result['Status']}")
                    st.markdown(f"**Processing Time:** {result.get('Processing Time', 'N/A')}")
                    st.markdown("---")
                    if result['Status'] == "Success":
                        # Escape dollar signs to prevent LaTeX rendering issues
                        st.markdown(result['Analysis'].replace("$", "\$"))
                    else:
                        st.error(result['Analysis'])
            else:
                st.info("We haven't analyzed anything yet. Go ahead and start an analysis when you're ready!")
        
        st.markdown("---")
    
    # Processing Section
    if files_to_process:
        st.header("Let's take a look")
        
        process_mode = st.radio(
            "How would you like to proceed?",
            ["Single File", "Batch Processing"],
            horizontal=True,
            help="Analyze one file at a time, or process all your files together?"
        )
        
        if st.button("Start analyzing", type="primary"):
            st.session_state.results = []
            
            if process_mode == "Single File":
                # Process first file only
                if len(files_to_process) > 0:
                    file = files_to_process[0]
                    file_name = file.name if hasattr(file, 'name') else "uploaded_file"
                    
                    # Create a copy to avoid file pointer issues
                    file.seek(0)
                    file_copy = BytesIO(file.read())
                    file_copy.name = file_name
                    
                    with st.spinner(f"Taking a close look at {file_name} for you..."):
                        start_time = time.time()
                        analysis = analyze_single_file(file_copy, file_name)
                        processing_time = time.time() - start_time
                        
                        status = "Success" if not analysis.startswith("Error") else "Failed"
                        st.session_state.results.append({
                            "File Name": file_name,
                            "Analysis": analysis,
                            "Status": status,
                            "Processing Time": f"{processing_time:.2f}s"
                        })
                        if status == "Success":
                            st.success(f"Great! We've finished analyzing {file_name}.")
            else:
                # Batch processing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(files_to_process):
                    file_name = file.name if hasattr(file, 'name') else f"file_{idx+1}"
                    status_text.text(f"Looking at {idx+1} of {len(files_to_process)}: {file_name}")
                    
                    # Create a copy to avoid file pointer issues
                    file.seek(0)
                    file_copy = BytesIO(file.read())
                    file_copy.name = file_name
                    
                    start_time = time.time()
                    analysis = analyze_single_file(file_copy, file_name)
                    processing_time = time.time() - start_time
                    
                    status = "Success" if not analysis.startswith("Error") else "Failed"
                    st.session_state.results.append({
                        "File Name": file_name,
                        "Analysis": analysis,
                        "Status": status,
                        "Processing Time": f"{processing_time:.2f}s"
                    })
                    
                    progress_bar.progress((idx + 1) / len(files_to_process))
                    time.sleep(1)  # Brief pause between files
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"Wonderful! We've finished analyzing all {len(files_to_process)} files for you.")
                st.rerun()
    
    # Results Section
    if st.session_state.results:
        st.header("Here's what we found")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        success_count = sum(1 for r in st.session_state.results if r["Status"] == "Success")
        total_count = len(st.session_state.results)
        with col1:
            st.metric("Files Analyzed", total_count)
        with col2:
            st.metric("Successful", f"{success_count}/{total_count}")
        with col3:
            avg_time = sum(float(r.get('Processing Time', '0').replace('s', '')) for r in st.session_state.results if 'Processing Time' in r)
            if total_count > 0:
                st.metric("Average Time", f"{avg_time/total_count:.2f}s")
        
        # Display results with side-by-side preview
        for idx, result in enumerate(st.session_state.results):
            with st.expander(
                f"{'✅' if result['Status'] == 'Success' else '❌'} {result['File Name']} "
                f"({result.get('Processing Time', 'N/A')})",
                expanded=(idx == 0)
            ):
                # Find corresponding uploaded file for preview
                file_preview = None
                files_to_check = uploaded_files if uploaded_files else st.session_state.uploaded_files_data
                for uploaded_file in files_to_check:
                    file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "unknown"
                    if file_name == result['File Name']:
                        file_preview = uploaded_file
                        break
                
                if file_preview:
                    col_preview, col_analysis = st.columns([1, 1])
                    with col_preview:
                        st.subheader("📊 Your Dashboard")
                        try:
                            # Create a copy to avoid file pointer issues
                            file_preview.seek(0)
                            file_copy = BytesIO(file_preview.read())
                            file_copy.name = result['File Name']
                            
                            image, error = load_image_from_file(file_copy, result['File Name'])
                            if not error:
                                st.image(image, caption=result['File Name'], use_container_width=True)
                            else:
                                st.warning("We couldn't load the preview, but the analysis is ready below.")
                        except Exception as e:
                            st.warning(f"Preview isn't available right now, but your analysis is ready.")
                    
                    with col_analysis:
                        st.subheader("📝 What We Discovered")
                        if result['Status'] == "Success":
                            st.markdown(result['Analysis'].replace("$", "\$"))
                        else:
                            st.error(f"Oops, we ran into an issue with this file: {result['Analysis']}")
                else:
                    # If file not found, just show analysis
                    if result['Status'] == "Success":
                        st.markdown(result['Analysis'].replace("$", "\$"))
                    else:
                        st.error(f"We encountered a problem: {result['Analysis']}")
        
        # Export Section
        st.header("Save your results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to Excel"):
                df = pd.DataFrame(st.session_state.results)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Analysis Results')
                output.seek(0)
                st.download_button(
                    label="⬇️ Download Excel file",
                    data=output,
                    file_name="dashboard_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📄 Export to CSV"):
                df = pd.DataFrame(st.session_state.results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV file",
                    data=csv,
                    file_name="dashboard_analysis.csv",
                    mime="text/csv"
                )
        
        # Clear results button
        if st.button("🗑️ Start fresh"):
            st.session_state.results = []
            st.session_state.uploaded_files_data = []
            st.rerun()


if __name__ == "__main__":
    main()


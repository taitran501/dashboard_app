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

# System instruction constants
SYS_INSTRUCTION_PERSONAL = """
You are a Senior Strategic Data Consultant. Your client is a C-Level Executive. 
You are analyzing a dashboard that may consist of multiple pages (e.g., Export, Import, Market Overview).

### 1. ANALYTICAL MINDSET
*   **Holistic View:** Do not analyze pages in isolation. Look for connections across pages (e.g., "Does the drop in Raw Material Imports on Page 2 explain the drop in Finished Goods Exports on Page 1?").
*   **The "So What?":** For every major trend, explain the business impact.
*   **Pareto Principle:** Focus heavily on the top 20% of drivers that create 80% of the value.
*   **Strict Honesty:** If data is unreadable, ambiguous, or missing, state "Data not actionable/visible."

### 2. REPORT STRUCTURE (Strict Markdown)

#### 🎯 Executive Bottom Line
*   **The Verdict:** A single, powerful sentence summarizing the overall business health.
*   **Critical KPI Snapshot:** The 3-4 most vital numbers (Total Revenue, Volume, Net Trade Balance) with a status (✅ On Track / ⚠️ At Risk).

#### 🔗 Supply Chain & Trade Dynamics
*   **Import vs. Export:** Compare inflows and outflows. Are we buying more than we are selling? 
*   **Margin/Value Check:** Compare Unit Prices of Imports vs. Exports. Are we adding sufficient value?
*   **Inventory Signals:** (e.g., "High imports but low exports suggests a stockpile buildup.")

#### 🧠 Strategic Insights & Drivers
*   **Top Performers:** Who are the key Buyers/Suppliers driving the business?
*   **Concentration Risk:** Are we too dependent on one client or supplier? (e.g., "Buyer A accounts for >50% of revenue").

#### 📉 Trends & Anomalies
*   **Red Flags:** Highlight sudden spikes, drops, or data gaps.

#### 💡 Actionable Recommendations
*   **Defensive Moves:** (e.g., "Diversify the supplier base to reduce reliance on Supplier X.")
*   **Growth Opportunities:** (e.g., "Expand sales in the North Region as it shows the highest ROI.")
"""

# Enterprise version: same structure but explicitly OMIT the Actionable Recommendations section
SYS_INSTRUCTION_ENTERPRISE = """
You are a Senior Strategic Data Consultant. Your client is a C-Level Executive. 
You are analyzing a dashboard that may consist of multiple pages (e.g., Export, Import, Market Overview).

### 1. ANALYTICAL MINDSET
*   **Holistic View:** Do not analyze pages in isolation. Look for connections across pages (e.g., "Does the drop in Raw Material Imports on Page 2 explain the drop in Finished Goods Exports on Page 1?").
*   **The "So What?":** For every major trend, explain the business impact.
*   **Pareto Principle:** Focus heavily on the top 20% of drivers that create 80% of the value.
*   **Strict Honesty:** If data is unreadable, ambiguous, or missing, state "Data not actionable/visible."

### 2. REPORT STRUCTURE (Strict Markdown)

#### 🎯 Executive Bottom Line
*   **The Verdict:** A single, powerful sentence summarizing the overall business health.
*   **Critical KPI Snapshot:** The 3-4 most vital numbers (Total Revenue, Volume, Net Trade Balance) with a status (✅ On Track / ⚠️ At Risk).

#### 🔗 Supply Chain & Trade Dynamics
*   **Import vs. Export:** Compare inflows and outflows. Are we buying more than we are selling? 
*   **Margin/Value Check:** Compare Unit Prices of Imports vs. Exports. Are we adding sufficient value?
*   **Inventory Signals:** (e.g., "High imports but low exports suggests a stockpile buildup.")

#### 🧠 Strategic Insights & Drivers
*   **Top Performers:** Who are the key Buyers/Suppliers driving the business?
*   **Concentration Risk:** Are we too dependent on one client or supplier? (e.g., "Buyer A accounts for >50% of revenue").

#### 📉 Trends & Anomalies
*   **Red Flags:** Highlight sudden spikes, drops, or data gaps.

IMPORTANT: Do NOT generate a separate \"Actionable Recommendations\" section. Focus only on insights, trends, and risks.
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
if 'uploaded_files_data_personal' not in st.session_state:
    st.session_state.uploaded_files_data_personal = []
if 'uploaded_files_data_enterprise' not in st.session_state:
    st.session_state.uploaded_files_data_enterprise = []
if 'models' not in st.session_state:
    # cache models per mode so we don't recreate them every time
    st.session_state.models = {}


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


def initialize_model(api_key, mode: str = "personal"):
    """Initialize Gemini model with given API key and mode (personal/enterprise)."""
    # Reuse cached model if available
    cached = st.session_state.models.get(mode)
    if cached is not None:
        return cached

    try:
        genai.configure(api_key=api_key)
        sys_instruction = SYS_INSTRUCTION_PERSONAL if mode == "personal" else SYS_INSTRUCTION_ENTERPRISE
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=sys_instruction,
        )
        st.session_state.models[mode] = model
        return model
    except Exception:
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

def load_content_from_file(file_input, file_name):
    """
    Load content from file.
    Returns: 
        - list of PIL Images (one per page if PDF)
        - error message (or None)
    """
    images = []
    try:
        if file_name.lower().endswith('.pdf'):
            # Handle PDF - Extract ALL pages
            file_input.seek(0)
            pdf_bytes = file_input.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            if len(doc) < 1:
                return None, "This PDF file appears to be empty"
            
            # Limit to first 10 pages to prevent overloading/timeouts
            pages_to_process = min(len(doc), 10)
            
            for i in range(pages_to_process):
                page = doc.load_page(i)
                
                # Zoom logic for better OCR resolution
                rect = page.rect
                target_width = 1600
                zoom = target_width / rect.width
                zoom = max(0.5, min(zoom, 2.0)) # Clamp zoom between 0.5x and 2.0x
                
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
            doc.close()
            return images, None
            
        else:
            # Handle standard image files (JPG, PNG)
            file_input.seek(0)
            img = Image.open(io.BytesIO(file_input.read()))
            images.append(img)
            return images, None
            
    except Exception as e:
        return None, f"Sorry, we couldn't read this file: {str(e)}"

def analyze_single_file(file_input, file_name, mode: str = "personal"):
    """Analyze a single file (PDF with multiple pages, or Image).

    mode: \"personal\" (full recommendations) or \"enterprise\" (no recommendations).
    """
    # Get API key and initialize model
    api_key = get_next_api_key()
    if not api_key:
        return "Oops! We couldn't access our API keys. Please check your key.txt file."
    
    model = initialize_model(api_key, mode=mode)
    if not model:
        return "Sorry, we had trouble connecting to our analysis service."
    
    # 1. Content Loading (Now returns a list of images)
    images, error = load_content_from_file(file_input, file_name)
    if error:
        return error

    # 2. Prepare content inputs
    content_inputs = []
    
    # Add prompt first or last, Gemini handles both. 
    # Adding a context prompt helps the model understand it might see multiple pages.
    user_prompt = f"""
    Analyze the attached dashboard images. This document contains {len(images)} page(s).
    
    If there are multiple pages (e.g. Export vs Import), compare them to find business correlations.
    Follow the 'Senior Strategic Data Consultant' system instruction structure strictly.
    """
    content_inputs.append(user_prompt)

    # 3. Add all images to the request
    for img in images:
        # Optional: Apply split_image_smart if an individual page is extremely tall
        # But usually PDF pages are standard ratio. We pass the whole page.
        # We resize if huge to save bandwidth/tokens
        w, h = img.size
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
        content_inputs.append(img)

    # 4. Inference with retry logic
    max_retries = len(st.session_state.api_keys)
    for attempt in range(max_retries):
        try:
            # Increase temperature slightly for more \"insightful/creative\" connections
            generation_config = genai.types.GenerationConfig(
                temperature=0.2, 
                max_output_tokens=8192,
            )

            response = model.generate_content(
                content_inputs,
                generation_config=generation_config
            )

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
                elif candidate.finish_reason == 3: # Safety block
                    if attempt < max_retries - 1:
                        api_key = get_next_api_key()
                        model = initialize_model(api_key)
                        continue
                    return "Safety filters blocked the analysis. Please check the file content."
            
            return "No response generated. Please try again."

        except Exception as e:
            error_msg = str(e)
            if ("quota" in error_msg.lower() or "rate" in error_msg.lower()) and attempt < max_retries - 1:
                api_key = get_next_api_key()
                model = initialize_model(api_key)
                time.sleep(1)
                continue
            return f"Analysis failed: {error_msg}"
            
    return "Connection failed after multiple attempts."

def main():
    # ------------------- 1. THEME SETUP -------------------
    # Initialize theme state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    # Define color palettes
    themes = {
        "light": {
            "bg": "#F8FAFC",             # Slate-50
            "text": "#1E293B",           # Slate-800
            "card": "#FFFFFF",           # White
            "border": "#E2E8F0",         # Slate-200
            "shadow": "rgba(0, 0, 0, 0.05)"
        },
        "dark": {
            "bg": "#0F172A",             # Slate-900
            "text": "#F8FAFC",           # Slate-50
            "card": "#1E293B",           # Slate-800
            "border": "#334155",         # Slate-700
            "shadow": "rgba(0, 0, 0, 0.3)"
        }
    }
    
    # Get current theme colors
    current = themes[st.session_state.theme]

    # ------------------- 2. DYNAMIC CSS INJECTION -------------------
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Dynamic Variables based on Theme */
    :root {{
        --background-color: {current['bg']};
        --text-color: {current['text']};
        --card-color: {current['card']};
        --border-color: {current['border']};
        --shadow-color: {current['shadow']};
        --primary-color: #4F46E5;
    }}

    /* Global Reset */
    * html, body, [class*="st-"], h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, button {{
        font-family: 'Inter', sans-serif !important;
        color: var(--text-color) !important;
    }}
    
    .stApp {{
        background-color: var(--background-color);
        transition: background-color 0.3s ease;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    [data-testid="stIconMaterial"] {{ display: none !important; }}

    /* Custom Headers */
    h1 {{
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        padding-bottom: 0.5rem;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2) !important;
        transition: transform 0.2s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
    }}

    /* Cards (Metrics/Uploaders) */
    [data-testid="stMetric"], [data-testid="stFileUploader"] {{
        background-color: var(--card-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px var(--shadow-color) !important;
    }}

    /* Inner text for uploader */
    [data-testid="stFileUploader"] * {{
        color: var(--text-color) !important;
    }}

    /* Selectboxes / dropdown inputs */
    .stSelectbox > div > div {{
        background-color: var(--card-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }}
    /* Dropdown menu panel */
    .stSelectbox [data-baseweb="popover"] {{
        background-color: var(--card-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }}
    /* Options inside dropdown */
    .stSelectbox [data-baseweb="menu"] div[role="option"] {{
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
    }}
    .stSelectbox [data-baseweb="menu"] div[role="option"]:hover {{
        background-color: rgba(79, 70, 229, 0.08) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--card-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 30px !important;
        padding: 5px 20px !important;
        color: var(--text-color) !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: rgba(79, 70, 229, 0.1) !important;
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ------------------- 3. THEME TOGGLE (VISIBLE IN MAIN) -------------------
    toggle_col, _ = st.columns([1, 4])
    with toggle_col:
        use_dark = st.toggle(
            "🌙 Dark mode",
            value=st.session_state.theme == "dark",
            help="Switch between light and dark themes",
        )
    if use_dark and st.session_state.theme != "dark":
        st.session_state.theme = "dark"
        st.rerun()
    elif not use_dark and st.session_state.theme != "light":
        st.session_state.theme = "light"
        st.rerun()

    # ------------------- 4. SIDEBAR (STATUS ONLY) -------------------
    with st.sidebar:
        st.markdown("### ⚙️ System")
        if not st.session_state.api_keys:
            st.session_state.api_keys = load_api_keys()
        if st.session_state.api_keys:
            st.success(f"**{len(st.session_state.api_keys)}** API keys loaded")
        else:
            st.error("No API keys found")

    # ------------------- 4. MAIN CONTENT -------------------
    # Hero Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1>Dashboard Insights AI</h1>
        <p style="opacity: 0.8;">Turn your flat dashboards into strategic intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Load API keys
    if not st.session_state.api_keys:
        st.session_state.api_keys = load_api_keys()
        if not st.session_state.api_keys:
            st.error("🔒 API configuration required. Please check your setup.")
            st.stop()
    
    st.markdown("---")    
    # Header with humanized copy
    st.title("Welcome! Let's analyze your dashboards together")
    st.markdown("Share your dashboard images with us, and we'll help you discover the insights that matter most. We support PDF, PNG, and JPG files.")
    
    # File Input Section
    st.header("Share your dashboards with us")

    tab_personal, tab_enterprise = st.tabs(["For you", "For enterprise"])

    def render_mode_section(mode_key: str, storage_key: str, friendly_title: str, teaser: str | None = None):
        """Render upload, preview, and processing for a given mode."""
        st.subheader(friendly_title)

        # Label outside of the uploader card for better spacing
        st.markdown("**Select your dashboard files**")
        uploads = st.file_uploader(
            "",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="You can upload one or more dashboard files at once. We'll analyze each one for you.",
            key=f"uploader_{mode_key}",
            label_visibility="collapsed",
        )

        # Normalize to list or empty list
        if uploads is None:
            uploads_list = []
        else:
            uploads_list = list(uploads)

        # When there are new uploads, overwrite stored files.
        if uploads_list:
            st.session_state[storage_key] = []
            for file in uploads_list:
                file.seek(0)
                file_data = BytesIO(file.read())
                file_data.name = file.name
                st.session_state[storage_key].append(file_data)
        # When user clears the uploader (no files), also clear stored cache for this mode.
        elif storage_key in st.session_state:
            st.session_state[storage_key] = []

        files_to_process = st.session_state.get(storage_key, [])

        # Preview Section for uploaded files
        if files_to_process:
            st.caption("Take a quick look at your dashboards before we dive into the insights.")

            preview_tab1, preview_tab2 = st.tabs(["📊 Your Dashboard", "📝 Your Results"])

            with preview_tab1:
                selected_file_idx = st.selectbox(
                    "Which dashboard would you like to see?",
                    range(len(files_to_process)),
                    format_func=lambda x: files_to_process[x].name if hasattr(files_to_process[x], 'name') else f"File {x+1}",
                    key=f"preview_select_{mode_key}",
                )

                if selected_file_idx is not None and selected_file_idx < len(files_to_process):
                    selected_file = files_to_process[selected_file_idx]
                    file_name = selected_file.name if hasattr(selected_file, 'name') else f"file_{selected_file_idx+1}"

                    try:
                        selected_file.seek(0)
                        file_copy = BytesIO(selected_file.read())
                        file_copy.name = file_name

                        images, error = load_content_from_file(file_copy, file_name)

                        if error:
                            st.error(error)
                        else:
                            st.image(
                                images[0],
                                caption=f"{file_name} (Page 1 of {len(images)})",
                                use_container_width=True,
                            )

                            if len(images) > 1:
                                with st.expander(f"View all {len(images)} pages"):
                                    for i, img in enumerate(images):
                                        st.image(
                                            img,
                                            caption=f"Page {i+1}",
                                            use_container_width=True,
                                        )
                    except Exception as e:
                        st.error(f"Error loading preview: {str(e)}")

            with preview_tab2:
                if st.session_state.results:
                    selected_result_idx = st.selectbox(
                        "Which result would you like to review?",
                        range(len(st.session_state.results)),
                        format_func=lambda x: st.session_state.results[x]['File Name'],
                        key=f"result_select_{mode_key}",
                    )

                    if selected_result_idx is not None and selected_result_idx < len(st.session_state.results):
                        result = st.session_state.results[selected_result_idx]
                        st.markdown(f"**File:** {result['File Name']}")
                        st.markdown(f"**Status:** {result['Status']}")
                        st.markdown(f"**Processing Time:** {result.get('Processing Time', 'N/A')}")
                        st.markdown(f"**Mode:** {result.get('Mode', 'personal').title()}")
                        st.markdown("---")
                        if result['Status'] == "Success":
                            st.markdown(result['Analysis'].replace("$", "\\$"))
                            if result.get("Mode") == "enterprise":
                                st.info("Want tailored action plans? Subscribe to unlock detailed recommendations.")
                        else:
                            st.error(result['Analysis'])
                else:
                    st.info("We haven't analyzed anything yet. Go ahead and start an analysis when you're ready!")

            st.markdown("---")

        # Processing Section (always batch over all files)
        if files_to_process:
            st.markdown("### Let's take a look")
            st.caption("We'll analyze all the dashboards you've uploaded in one go.")

            if st.button("Start analyzing", type="primary", key=f"start_analyzing_{mode_key}"):
                st.session_state.results = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, file in enumerate(files_to_process):
                    file_name = file.name if hasattr(file, 'name') else f"file_{idx+1}"
                    status_text.text(f"Looking at {idx+1} of {len(files_to_process)}: {file_name}")

                    file.seek(0)
                    file_copy = BytesIO(file.read())
                    file_copy.name = file_name

                    start_time = time.time()
                    analysis = analyze_single_file(file_copy, file_name, mode=mode_key)
                    processing_time = time.time() - start_time

                    status = "Success" if not analysis.startswith("Error") else "Failed"
                    st.session_state.results.append({
                        "File Name": file_name,
                        "Analysis": analysis,
                        "Status": status,
                        "Processing Time": f"{processing_time:.2f}s",
                        "Mode": mode_key,
                    })

                    progress_bar.progress((idx + 1) / len(files_to_process))
                    time.sleep(1)

                status_text.empty()
                progress_bar.empty()
                st.success(f"Wonderful! We've finished analyzing all {len(files_to_process)} files for you.")

    with tab_personal:
        render_mode_section(
            mode_key="personal",
            storage_key="uploaded_files_data_personal",
            friendly_title="For you – get full insights plus actionable recommendations",
        )

    with tab_enterprise:
        render_mode_section(
            mode_key="enterprise",
            storage_key="uploaded_files_data_enterprise",
            friendly_title="For enterprise – high-level insights (no detailed recommendations)",
        )
    
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
                files_to_check = (
                    st.session_state.get("uploaded_files_data_personal", [])
                    + st.session_state.get("uploaded_files_data_enterprise", [])
                )
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
                            
                            images, error = load_content_from_file(file_copy, result['File Name'])
                            if not error and images:
                                st.image(
                                    images[0],
                                    caption=f"{result['File Name']} (Page 1 of {len(images)})",
                                    use_container_width=True,
                                )
                            else:
                                st.warning("We couldn't load the preview, but the analysis is ready below.")
                        except Exception as e:
                            st.warning(f"Preview isn't available right now, but your analysis is ready.")
                    
                    with col_analysis:
                        st.subheader("📝 What We Discovered")
                        if result['Status'] == "Success":
                            st.markdown(result['Analysis'].replace("$", "\\$"))
                            if result.get("Mode") == "enterprise":
                                st.info("Want tailored action plans? Subscribe to unlock detailed recommendations.")
                        else:
                            st.error(f"Oops, we ran into an issue with this file: {result['Analysis']}")
                else:
                    # If file not found, just show analysis
                    if result['Status'] == "Success":
                        st.markdown(result['Analysis'].replace("$", "\\$"))
                        if result.get("Mode") == "enterprise":
                            st.info("Want tailored action plans? Subscribe to unlock detailed recommendations.")
                    else:
                        st.error(f"We encountered a problem: {result['Analysis']}")
        
        # Export Section
        st.header("Save your results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to Excel"):
                df = pd.DataFrame(st.session_state.results)
                # Drop internal columns we don't want to expose
                df = df.drop(columns=["Processing Time", "Status"], errors="ignore")
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
                df = df.drop(columns=["Processing Time", "Status"], errors="ignore")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV file",
                    data=csv,
                    file_name="dashboard_analysis.csv",
                    mime="text/csv"
                )
        
        # Clear results button
        if st.button("🗑️ Clear history"):
            st.session_state.results = []
            st.session_state.uploaded_files_data = []
            st.rerun()


if __name__ == "__main__":
    main()


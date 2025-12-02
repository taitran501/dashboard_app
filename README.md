# Dashboard Analyzer - Streamlit Web App

A web application for analyzing business dashboard images (PDF, PNG, JPG) using Google Gemini AI to provide executive summaries and insights.

## Features

- 📊 Analyze dashboard images using Google Gemini 2.5 Flash
- 🔄 Automatic API key rotation for rate limit management
- 📁 Support for multiple file formats (PDF, PNG, JPG)
- ⚡ Single file and batch processing modes
- 📋 View results in expandable sections
- 💾 Export results to Excel or CSV
- 🎯 Executive summary format with insights

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

The app supports multiple methods for API key configuration (in priority order):

**Option 1: .env file (Recommended)**
Create a `.env` file in the root directory:

```
api_key_1=your_api_key_1_here
api_key_2=your_api_key_2_here
api_key_3=your_api_key_3_here
...
api_key_8=your_api_key_8_here
```

**Option 2: key.txt file**
Create a `key.txt` file in the root directory:

```
api_key_1="your_api_key_1_here"
api_key_2="your_api_key_2_here"
api_key_3="your_api_key_3_here"
...
```

**Option 3: Environment Variables**
Set environment variables:

```bash
export GEMINI_API_KEY_1="your_api_key_1"
export GEMINI_API_KEY_2="your_api_key_2"
...
```

The app will automatically rotate through all available keys to avoid rate limits.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload Files**: Use the "Upload Files" tab to select dashboard files from your computer
2. **Select Processing Mode**: Choose between single file or batch processing
3. **Start Analysis**: Click "Start Analysis" to process your files
4. **View Results**: Results are displayed in expandable sections with markdown formatting
5. **Export**: Download results as Excel or CSV files

## File Structure

```
dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── key.txt               # API keys (not in git)
├── .env.example          # Example environment variables
├── README.md             # This file
└── data/                 # Optional: folder for dashboard files
```

## API Key Rotation

The application automatically rotates through all available API keys:
- Each request uses the next key in rotation
- If a key fails (rate limit, error), the app automatically tries the next key
- Rotation helps distribute load and avoid rate limits

## Supported File Formats

- **PDF**: First page is extracted and analyzed
- **PNG**: Direct image analysis
- **JPG/JPEG**: Direct image analysis

## Output Format

The analysis provides:
- 🎯 Executive Snapshot (one-sentence verdict + primary metrics)
- 📊 Critical Drivers (top performers + concentration risk)
- 📈 Trend & Timeline Story (trajectory, movements, seasonality)
- 🚨 Anomalies & Red Flags (outliers + data gaps)

## Notes

- Large images are automatically resized to fit API limits
- Very tall images are split into multiple parts
- Processing time depends on image complexity and API response time
- Results are stored in session state and can be exported

## Troubleshooting

- **No API keys found**: Ensure `key.txt` exists with at least one API key
- **API errors**: Check your API key validity and quota
- **File read errors**: Ensure files are valid PDF/PNG/JPG formats
- **Rate limits**: The app will automatically rotate to next key


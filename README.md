# MindLens - AI-Powered Digital Diary 🧠✨

**MindLens** is your private, intelligent digital diary that understands your emotions, thoughts, and mental well-being. Using advanced AI models running **100% locally on your device**, MindLens provides deep insights into your emotional journey while keeping your privacy intact.

---

## 🌟 Key Features Overview

### 🎭 **AI-Powered Emotion Detection**
Automatically analyzes your diary entries to identify emotions using a fine-tuned BERT model trained on the GoEmotions dataset. Detects multiple emotions simultaneously including joy, sadness, anxiety, excitement, and 25+ other emotions.

### 🏷️ **Intelligent Context Tagging**
Identifies topics and life contexts in your writing using a custom-trained SpaCy model. Automatically categorizes entries into 20+ context tags including work, family, health, relationships, creativity, productivity, and more.

### 🔍 **Semantic Search Engine**
Find past entries using natural language queries powered by FAISS vector search and sentence embeddings. Search by meaning, not just keywords - ask questions like "when did I feel proud?" or "stressful work moments".

### 🤖 **AI-Powered Summaries (RAG)**
Get intelligent insights about patterns in your thoughts and feelings using Retrieval-Augmented Generation (RAG) with Hugging Face's Llama 3.1 model. Ask questions like "How have I been feeling about my career lately?" and receive thoughtful, context-aware summaries.

### 🎤 **Audio Transcription**
Speak your thoughts instead of typing! Upload audio files (WAV, MP3, M4A) and MindLens will automatically transcribe them to text using the Whisper ASR model - all processed locally on your device.

### 🎥 **Video Transcription**
Capture video journals! Upload videos (MP4, AVI, MOV, MKV) and MindLens extracts the audio using FFmpeg, then transcribes it with Whisper. Perfect for video diaries or voice notes.

### 🖼️ **Image Attachment & Retrieval**
Attach photos to your diary entries with searchable descriptions. Later, retrieve entries by image description to find specific memories visually.

### 🛡️ **Mental Health Monitoring**
Advanced suicide risk detection using rule-based analysis identifies concerning content and provides:
- Real-time risk assessment with confidence scores
- Emotional state analysis
- Crisis intervention resources
- Automatic emergency contact alerts

### 📱 **WhatsApp Emergency Alerts**
When high-risk content is detected, MindLens can automatically send WhatsApp alerts to your pre-configured emergency contacts, ensuring someone knows when you might need support.

### ❓ **Comprehensive FAQ System**
Get instant help with our built-in Frequently Asked Questions system. Access 60+ detailed questions organized into 9 categories covering:
- Getting started and installation
- Using diary features (text, audio, video, images)
- Search and discovery capabilities
- AI features and models
- Mental health and safety features
- Analytics and insights
- Configuration and setup
- Troubleshooting common issues
- Privacy and security

Search for specific questions or browse by category. Each question expands to show detailed, helpful answers.

### 📊 **Interactive Analytics Dashboard**
Visualize your emotional journey with:
- Sentiment trends over time
- Emotion frequency charts
- Tag distribution analysis
- Daily emotion intensity heatmaps

### 📄 **PDF Export**
Export your diary entries as beautifully formatted PDF documents, complete with metadata, emotions, and tags.

### 🔒 **Privacy-First Design**
All AI processing happens **100% locally** on your device. No cloud APIs for core features. Your thoughts stay private. Optional cloud LLM for summaries only (can be disabled).

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+** installed
- **4GB+ RAM** recommended for model loading
- **FFmpeg** (optional, for video transcription)
- **Windows/Linux/Mac** operating system

### 1️⃣ Installation

#### Option A: Automated Setup (Recommended)
```bash
# Windows
run_setup.bat

# Linux/Mac
python setup.py
```

#### Option B: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download Whisper model (optional, for audio/video transcription)
python download_whisper_small.py
```

### 2️⃣ Configuration

Edit [`config.py`](c:\Users\gowth\Downloads\MINDLENS_WINDOWS_VERSION\MINDLENS\config.py) to customize settings:

```python
# Vector store backend: 'faiss' or 'chroma'
VECTOR_STORE = "faiss"

# LLM backend: 'huggingface' or 'none'
LLM_BACKEND = "huggingface"  # Set to 'none' for 100% offline operation

# Hugging Face API settings (for AI summaries)
HF_API_TOKEN = "your_token_here"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Emergency alert settings
AUTO_ALERT_ENABLED = True  # Enable/disable automatic WhatsApp alerts
```

### 3️⃣ First-Time Setup: Emergency Contact

When you first launch MindLens, you'll be prompted to add an emergency contact (name and phone number). This contact will only be notified in crisis situations when high-risk content is detected.

### 4️⃣ Launch MindLens

#### Web Interface (Recommended)
```bash
# Windows
start.bat
# or
streamlit run app.py

# Linux/Mac
python -m streamlit run app.py
```

The web interface will open in your browser at `http://localhost:8501`

### Web Interface Navigation

The MindLens web interface features an intuitive sidebar navigation system:

1. **📝 Add New Entry** - Create new diary entries with text, audio, video, or images
2. **🔍 Search Entries** - Find past entries using semantic search
3. **🤖 AI Summary** - Get AI-powered insights about your entries
4. **📊 Statistics** - View analytics and emotional trends
5. **📚 View All Entries** - Browse all your diary entries
6. **📄 Download PDF** - Export entries as PDF documents
7. **🛡️ Mental Support** - Access mental health resources and emergency contacts
8. **❓ FAQ** - Get help with common questions and troubleshooting
9. **ℹ️ About** - View application information and version details

All features are accessible through the left sidebar navigation panel.

#### Command Line Interface
```bash
python main.py
```

---

## 📖 How It Works

### System Architecture

MindLens uses a sophisticated multi-model AI pipeline:

```
┌─────────────────┐
│  Your Diary     │
│  Text/Audio/    │
│  Video Entry    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Preprocessing & Transcription          │
│  - Audio → Whisper ASR Model           │
│  - Video → FFmpeg + Whisper            │
│  - Text → Direct Processing            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Parallel AI Analysis (4 Models)        │
│  ┌────────────────────────────────┐    │
│  │ 1. Emotion Detection (BERT)    │    │
│  │    - Multi-label classification │    │
│  │    - 28 GoEmotions categories  │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │ 2. Context Tagging (SpaCy)     │    │
│  │    - Topic identification      │    │
│  │    - 20+ life context tags     │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │ 3. Embedding (SBERT)           │    │
│  │    - Semantic vector (384-dim) │    │
│  │    - For similarity search     │    │
│  └────────────────────────────────┘    │
│  ┌────────────────────────────────┐    │
│  │ 4. Suicide Risk (Rule-based)   │    │
│  │    - Risk score calculation    │    │
│  │    - Crisis detection          │    │
│  └────────────────────────────────┘    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Storage & Indexing                     │
│  - FAISS Vector Index (for search)      │
│  - JSONL metadata (emotions, tags)      │
│  - Local file storage (images, videos)  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Query & Retrieval                      │
│  - Semantic search via embeddings       │
│  - Filter by emotions/tags              │
│  - RAG-powered AI summaries             │
└─────────────────────────────────────────┘
```

### AI Models Used

1. **Emotion Detection**: Fine-tuned BERT model (`BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT/`)
   - Multi-label classification
   - 28 emotion categories from GoEmotions dataset
   - Threshold-based filtering for relevant emotions

2. **Context Tagging**: Custom SpaCy transformer model (`spacy_model_context/`)
   - Multi-label text categorization
   - 20+ life context categories
   - Keyword fallback for robustness

3. **Semantic Embeddings**: Sentence-BERT (`bert_model_offilne/`)
   - Model: `all-MiniLM-L6-v2`
   - 384-dimensional sentence embeddings
   - Cosine similarity for semantic search

4. **Audio Transcription**: Whisper Small (`models/small/`)
   - Automatic speech recognition
   - Multi-language support
   - Runs locally via faster-whisper

5. **Suicide Risk Detection**: Rule-based analyzer (`suicide_detector_fallback.py`)
   - 100+ risk indicators
   - Weighted scoring algorithm
   - Emotion and tag extraction

6. **AI Summaries**: Llama 3.1 8B Instruct (via Hugging Face API)
   - Retrieval-Augmented Generation (RAG)
   - Context-aware responses
   - Optional - can be disabled for full offline operation

---

## 🎯 Feature Details & Usage

### 1. Add New Entry

**How to use:**
1. Navigate to the main page or click **"📝 Add New Entry"**
2. Select the date for your entry
3. Write your thoughts in the text area, OR
4. Upload audio/video for automatic transcription
5. (Optional) Attach an image with a searchable description
6. Click **"✨ Save My Memory ✨"**

**What happens:**
- Text is analyzed for emotions (joy, sadness, anxiety, etc.)
- Context tags are automatically identified (work, family, health, etc.)
- Sentiment is calculated (positive/neutral/negative)
- Mental health risk is assessed
- Entry is embedded and stored in vector database
- If high risk detected: Emergency alert workflow activates

**Example:**
```python
from main import MindLensApp

app = MindLensApp()
result = app.process_entry(
    text="Had an amazing day! Completed my project ahead of schedule and celebrated with friends.",
    date="2025-10-29",
    image_path="data/images/celebration.jpg",
    image_desc="team celebration at office"
)

print(result)
# Output:
# {
#   'doc_id': 'abc123...',
#   'sentiment': 'positive',
#   'emotions': ['joy', 'excitement', 'pride'],
#   'tags': ['work', 'productivity', 'achievement', 'relationships'],
#   'suicide_score': 0.05,
#   'suicide_prediction': 'Non-Suicidal'
# }
```

### 2. Search Entries

**How to use:**
1. Click **"🔍 Search Entries"**
2. Enter a natural language query (e.g., "anxious moments at work")
3. (Optional) Filter by specific emotions or tags
4. Adjust number of results (1-20)
5. Click **"🔍 Search"**

**Search capabilities:**
- **Semantic search**: Finds entries by meaning, not just keywords
- **Emotion filtering**: Only show entries with specific emotions
- **Tag filtering**: Filter by life contexts (work, family, health, etc.)
- **Image retrieval**: Search for entries by image description

**Example:**
```python
results = app.search_entries(
    query="times I felt accomplished",
    top_k=10,
    filter_emotions=["pride", "joy"],
    filter_tags=["work", "achievement"]
)

for entry in results:
    print(f"[{entry['date']}] {entry['text'][:100]}...")
    print(f"Emotions: {', '.join(entry['emotions'])}")
```

### 3. AI Summary (RAG Chat)

**How to use:**
1. Click **"🤖 AI Summary"**
2. Ask a question about your diary (e.g., "What patterns do I have with stress?")
3. (Optional) Filter by tags or emotions
4. Adjust number of entries to analyze (1-20)
5. Click **"Search & Summarize"**

**What you get:**
- AI-powered analysis of your emotional patterns
- Insights about recurring themes
- Temporal analysis of mood changes
- Empathetic, thoughtful responses

**Example queries:**
- "How have I been feeling about my career lately?"
- "What makes me anxious?"
- "When was I happiest this month?"
- "What are my stress triggers?"

### 4. Statistics Dashboard

**Visualizations:**
- **Sentiment Over Time**: Line graph showing daily average sentiment
- **Emotion Frequency**: Bar chart of most common emotions
- **Tag Distribution**: Pie chart of life context categories
- **Emotion Heatmap**: Calendar-like view of emotional intensity

### 5. Mental Health Support

**Features:**
- **Crisis Resources**: National helplines and emergency contacts
- **Hospital Finder**: OpenStreetMap-based mental health facility locator
- **Emergency Contacts Manager**: Add/remove WhatsApp alert recipients
- **Crisis Intervention**: Automatic redirect when high risk detected

**Emergency Contact System:**
```python
from mental_health_service import mental_health_service

# Add emergency contact
mental_health_service.add_emergency_contact(
    name="Dr. Sarah Johnson",
    phone="+919876543210"
)

# Check if contacts exist
has_contacts = mental_health_service.has_emergency_contacts()
```

### 6. Download PDF

**Export your diary as a PDF:**
1. Click **"📄 Download PDF"**
2. Select date range (start and end dates)
3. (Optional) Filter by specific emotions or tags
4. Click **"📥 Generate PDF"**
5. Download the formatted PDF document

**PDF includes:**
- Formatted diary entries with dates
- Emotions and sentiment indicators
- Context tags
- Professional layout using ReportLab

---

## 🛡️ Mental Health & Safety Features

### Suicide Risk Detection

MindLens uses a sophisticated rule-based analysis system to identify concerning content:

**Risk Assessment Algorithm:**
1. **Keyword Analysis**: Scans for 100+ high-risk and medium-risk keywords
2. **Weighted Scoring**: High-risk keywords (4x weight), medium-risk (2x weight)
3. **Positive Counterbalance**: Positive keywords reduce risk score
4. **Threshold Classification**:
   - Score ≥ 0.7: High Risk
   - Score 0.4-0.7: Moderate Risk
   - Score < 0.4: Low Risk

**When High Risk is Detected:**

1. **Immediate Visual Alert**: Red banner with crisis resources
2. **Mental Health Analysis Card**: Shows risk level, confidence, emotional state
3. **Crisis Helplines**: National Suicide Prevention Lifeline (988), Crisis Text Line
4. **Redirect to Mental Support**: User can immediately access help resources
5. **WhatsApp Emergency Alerts** (if enabled):
   - Automatic notification to emergency contacts
   - Message: "🚨 CRISIS ALERT: High suicide risk detected in MindLens diary entry"
   - Sent only once per high-risk detection (no duplicates)
   - User can disable auto-alerts in config

**Privacy Note**: Mental health data stays 100% local. No external transmission.

### Emergency Contact Configuration

Edit emergency contacts in [`data/emergency_contacts.json`](c:\Users\gowth\Downloads\MINDLENS_WINDOWS_VERSION\data\emergency_contacts.json):

```json
[
  {
    "name": "Dr. Sarah Johnson (Therapist)",
    "phone": "+919876543210"
  },
  {
    "name": "Mom",
    "phone": "+919123456789"
  }
]
```

**WhatsApp Alert Requirements:**
- WhatsApp Web must be logged in on your browser
- Phone numbers must include country code (e.g., +91 for India)
- Uses `pywhatkit` library for browser automation
- Each alert opens a new browser tab (closes automatically)

---

## 🔧 Technical Specifications

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models + space for diary data
- **Optional**: FFmpeg for video transcription

### Dependencies

**Core ML Libraries:**
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
spacy>=3.7.0
spacy-transformers>=1.3.0
scikit-learn>=1.3.0
numpy>=1.24.0,<2.0.0
nltk>=3.8.0
```

**Vector Storage:**
```
faiss-cpu>=1.7.4
chromadb>=0.4.0
```

**LLM Integration:**
```
huggingface-hub>=0.17.0
```

**Web Framework:**
```
streamlit>=1.25.0
faster-whisper>=1.0.0
```

**Utilities:**
```
pandas>=2.0.0
plotly>=5.0.0
reportlab>=4.0.0
requests>=2.31.0
pywhatkit>=5.4
python-dotenv>=1.0.0
```

### Data Storage Structure

```
data/
├── faiss_index.faiss          # Vector index for semantic search
├── faiss_meta.jsonl           # Metadata (emotions, tags, text)
├── entries.jsonl              # Complete diary entries
├── emergency_contacts.json    # Emergency WhatsApp contacts
├── images/                    # Uploaded images
│   └── YYYYMMDD_HHMMSS_*.jpg
└── videos/                    # Uploaded videos
    └── YYYYMMDD_HHMMSS_*.mp4
```

### Performance Benchmarks

| Operation | Time (avg) | Notes |
|-----------|------------|-------|
| Add entry (text only) | 2-3 seconds | Includes all 4 AI models |
| Add entry with audio (1 min) | 15-20 seconds | Includes Whisper transcription |
| Add entry with video (1 min) | 20-30 seconds | FFmpeg + Whisper |
| Search entries | <1 second | FAISS vector search |
| AI summary generation | 5-10 seconds | Using HF API (varies by model) |
| PDF export (100 entries) | 3-5 seconds | |

---

## 🔒 Privacy & Security

### Data Privacy Guarantees

✅ **100% Local Processing** for core features:
- Emotion detection (BERT) - runs on your device
- Context tagging (SpaCy) - runs on your device  
- Embeddings (SBERT) - runs on your device
- Suicide risk detection - runs on your device
- Audio/video transcription (Whisper) - runs on your device
- Vector search (FAISS) - runs on your device

⚠️ **Optional Cloud Features** (can be disabled):
- AI summaries via Hugging Face API
  - Only summary queries are sent (not full diary)
  - Set `LLM_BACKEND = "none"` to disable
- WhatsApp alerts via browser automation
  - Uses local WhatsApp Web (no external API)
  - Requires your browser and WhatsApp session

### Security Best Practices

1. **API Keys**: Store in environment variables or `.env` file (never commit to Git)
2. **Emergency Contacts**: Keep `emergency_contacts.json` private
3. **Backup**: Regularly backup `data/` folder
4. **Access Control**: MindLens runs on localhost - not exposed to network

### Data Retention

- **Diary entries**: Stored indefinitely in local JSONL files
- **Images/Videos**: Stored indefinitely in local folders
- **Vector index**: Persists across sessions
- **No automatic deletion**: You control your data

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
**Error**: `FileNotFoundError: Model directory not found`

**Solution**:
- Verify model paths in [`config.py`](c:\Users\gowth\Downloads\MINDLENS_WINDOWS_VERSION\MINDLENS\config.py)
- Ensure all model directories exist:
  - `BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT/`
  - `spacy_model_context/`
  - `bert_model_offilne/`
  - `models/small/` (Whisper)

#### 2. Memory Errors
**Error**: `RuntimeError: CUDA out of memory` or system slowdown

**Solution**:
- Close other applications
- Models run on CPU by default (slower but works on any device)
- Ensure 4GB+ RAM available
- Consider processing entries one at a time

#### 3. FFmpeg Not Found (Video Transcription)
**Error**: `FileNotFoundError: ffmpeg not found`

**Solution**:
```bash
# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH

# Mac
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

Restart application after installing FFmpeg.

#### 4. WhatsApp Alerts Not Working
**Error**: Alerts not being sent

**Solution**:
- Ensure WhatsApp Web is logged in on your browser
- Verify phone numbers include country code (e.g., +91)
- Check `emergency_contacts.json` format is correct
- Allow browser automation (don't block pop-ups)

#### 5. Streamlit Port Already in Use
**Error**: `Port 8501 already in use`

**Solution**:
```bash
# Windows
taskkill /F /IM streamlit.exe

# Linux/Mac
kill -9 $(lsof -t -i:8501)

# Or run on different port
streamlit run app.py --server.port 8502
```

#### 6. Import Errors
**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Debug Mode

Enable verbose logging:
```bash
# Set environment variable
export MINDLENS_DEBUG=1  # Linux/Mac
set MINDLENS_DEBUG=1     # Windows

streamlit run app.py
```

---

## 🤝 Contributing

This is a personal privacy-focused project. If you find bugs or have suggestions:
1. Check existing issues
2. Verify the issue in latest version
3. Provide detailed error messages and steps to reproduce

---

## 📝 License

This project is for **personal use only**. 

Please respect the licenses of underlying models and libraries:
- BERT models: Apache 2.0
- SpaCy: MIT License
- Sentence-Transformers: Apache 2.0
- Whisper: MIT License
- Streamlit: Apache 2.0

---

## 🙏 Acknowledgments

- **GoEmotions Dataset**: Fine-grained emotion classification
- **Hugging Face**: Transformers and model hosting
- **SpaCy**: Industrial-strength NLP
- **OpenAI**: Whisper ASR model
- **FAISS**: Efficient similarity search by Meta AI
- **Streamlit**: Beautiful web apps for ML

---

## 📞 Support & Resources

### Mental Health Resources (24/7)

🇺🇸 **United States**
- National Suicide Prevention Lifeline: **988**
- Crisis Text Line: Text **HOME** to **741741**
- Emergency: **911**

🇮🇳 **India**
- KIRAN Mental Health Helpline: **1800-599-0019** (Toll-free)
- Vandrevala Foundation: **9999 666 555**
- Suicide Prevention India: **9820466726**
- Emergency: **100**

### Technical Support

For technical issues:
1. Check [Troubleshooting](#-troubleshooting) section
2. Verify all dependencies are installed: `pip list`
3. Check model paths in `config.py`
4. Ensure sufficient system resources (RAM, disk space)
5. Review error logs in console output

---

## 🔮 Future Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] End-to-end encryption for backups
- [ ] Multi-language support (currently English)
- [ ] Voice mood analysis (pitch, tone detection)
- [ ] Integration with wearable devices (heart rate, sleep data)
- [ ] Collaborative journaling (shared diaries)
- [ ] Advanced visualizations (emotion maps, word clouds)
- [ ] Plugin system for extensibility

---

**Made with ❤️ for mental health awareness and personal growth**

*Remember: Your mental health matters. You are not alone.*

#!/usr/bin/env python3
"""
Streamlit web interface for MindLens.
"""

import os
# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, date
import json
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import with error handling
try:
    from main import MindLensApp
    from config import APP_TITLE, ABOUT_TEXT, WHISPER_MODEL_DIR, IMAGES_DIR, VIDEOS_DIR
    from mental_health_service import mental_health_service
    APP_AVAILABLE = True
except Exception as e:
    st.error(f"⚠️ Application initialization error: {e}")
    st.info("Please restart the application to resolve this issue.")
    APP_AVAILABLE = False
    APP_TITLE = "MindLens - AI-Powered Digital Diary"
    ABOUT_TEXT = "Your AI-powered digital diary with emotion detection and intelligent insights."

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        border-left: 6px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    
    .metric-card h3 {
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.2;
        height: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: bold;
        line-height: 1.2;
        height: 1.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        word-wrap: break-word;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.2;
        height: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1.6rem;
        font-weight: bold;
        line-height: 1.2;
        height: 1.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        word-wrap: break-word;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .emotion-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .tag-badge {
        background: linear-gradient(45deg, #48cae4, #023e8a);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .sidebar-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 0.8rem;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sidebar-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .success-message {
        background: linear-gradient(45deg, #00b894, #00cec9);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-size: 1.2rem;
        text-align: center;
    }
    
    .search-result {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .summary-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .entry-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border-left: 6px solid #48cae4;
    }
    
    .download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .delete-warning {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .delete-info {
        background: linear-gradient(45deg, #ffa726, #ffb74d);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Custom hover effects for gradient cards */
    .gradient-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .gradient-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Enhanced file uploader styling */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #e0e0e0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    /* Enhanced text input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced textarea styling */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Navigation Buttons */
    .stSidebar .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        text-align: left;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        margin-bottom: 0.5rem;
    }
    
    .stSidebar .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateX(5px) translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stSidebar .stButton > button:active {
        transform: translateX(3px) translateY(0px);
    }
    
    /* Smooth animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-card, .metric-card, .gradient-card {
        animation: fadeIn 0.5s ease;
    }

    /* 🔥 Hide Streamlit's default multipage navigation */
    section[data-testid="stSidebarNav"] {display: none;}
    div[data-testid="stSidebarNav"] {display: none;}
```
""", unsafe_allow_html=True)


# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_app():
    """Initialize the MindLens app."""
    if not st.session_state.initialized:
        if not APP_AVAILABLE:
            st.error("⚠️ Application not available due to initialization error.")
            st.info("Please restart the application to resolve this issue.")
            return
            
        with st.spinner("Initializing MindLens..."):
            try:
                st.session_state.app = MindLensApp()
                st.session_state.initialized = True
                st.success("MindLens initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize MindLens: {e}")
                st.stop()

def prompt_for_emergency_contact():
    """Prompt user for emergency contact if none exist."""
    from mental_health_service import mental_health_service
    
    # Check if emergency contacts exist
    if not mental_health_service.has_emergency_contacts():
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            border: 2px solid #ff6b6b;
        ">
            <h2 style="color: white; margin: 0 0 1rem 0;">⚠️ Emergency Contact Required</h2>
            <p style="color: white; font-size: 1.1rem; margin: 0 0 1.5rem 0;">
                Please add an emergency contact for your safety. This contact will only be notified 
                in crisis situations when high-risk content is detected.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a form for emergency contact
        with st.form("initial_emergency_contact"):
            st.markdown("### 📞 Add Your Emergency Contact")
            st.markdown("This contact will only be notified in crisis situations.")
            
            col1, col2 = st.columns(2)
            with col1:
                contact_name = st.text_input("Contact Name", placeholder="e.g., Parent, Friend, Therapist")
            with col2:
                contact_phone = st.text_input("Phone Number", placeholder="e.g., +919876543210 or 9876543210")
            
            st.markdown("*Your privacy is important - contact information is stored locally and never shared.*")
            
            submitted = st.form_submit_button("➕ Add Emergency Contact", type="primary")
            
            if submitted:
                if contact_name.strip() and contact_phone.strip():
                    # Add contact using mental_health_service
                    if mental_health_service.add_emergency_contact(contact_name.strip(), contact_phone.strip()):
                        st.success(f"✅ Emergency contact '{contact_name}' added successfully!")
                        st.info("You can add more contacts later in the Mental Support section.")
                        st.session_state.emergency_contact_added = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to add emergency contact. Please try again.")
                else:
                    st.warning("⚠️ Please enter both name and phone number.")

def main():
    """Main Streamlit app."""
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🧠 {APP_TITLE}</h1>
        <p>Your AI-powered digital diary with emotion detection and intelligent insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize app
    initialize_app()
    
    # Check if we need to prompt for emergency contact (only if app is initialized)
    if st.session_state.initialized:
        # Initialize emergency contact prompt state
        if 'emergency_contact_added' not in st.session_state:
            st.session_state.emergency_contact_added = False
        
        # Only prompt if no emergency contact has been added yet
        if not st.session_state.emergency_contact_added:
            from mental_health_service import mental_health_service
            if not mental_health_service.has_emergency_contacts():
                prompt_for_emergency_contact()
                return  # Don't show the rest of the app until emergency contact is added
    
    # Sidebar with separate buttons
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        ">
            <h3 style="
                color: white;
                margin: 0;
                font-size: 1.3rem;
                font-weight: 600;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            ">🧭 Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        if st.button("📝 Add New Entry", key="nav_add", use_container_width=True):
            st.session_state.current_page = "Add Entry"
        
        if st.button("🔍 Search Entries", key="nav_search", use_container_width=True):
            st.session_state.current_page = "Search Entries"
        
        if st.button("🤖 AI Summary", key="nav_summary", use_container_width=True):
            st.session_state.current_page = "AI Summary"
        
        if st.button("📊 Statistics", key="nav_stats", use_container_width=True):
            st.session_state.current_page = "Statistics"
        
        if st.button("📚 View All Entries", key="nav_entries", use_container_width=True):
            st.session_state.current_page = "View Entries"
        
        if st.button("📄 Download PDF", key="nav_download", use_container_width=True):
            st.session_state.current_page = "Download PDF"
        
        if st.button("🛡️ Mental Support", key="nav_mental", use_container_width=True):
            st.session_state.current_page = "Mental Support"
        
        if st.button("❓ FAQ", key="nav_faq", use_container_width=True):
            st.session_state.current_page = "FAQ"
        
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.current_page = "About"
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        if st.session_state.initialized:
            stats = st.session_state.app.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🗄️ Vector Store", stats['vector_store'].title())
            with col2:
                st.metric("🤖 LLM Backend", stats['llm_backend'].title())
            
            # Model status
            st.markdown("#### 🧠 Model Status")
            models = stats['models_loaded']
            for model, status in models.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {model.replace('_', ' ').title()}")
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Add Entry"
    
    # Main content area
    if st.session_state.current_page == "Add Entry":
        add_entry_page()
    elif st.session_state.current_page == "Search Entries":
        search_entries_page()
    elif st.session_state.current_page == "AI Summary":
        ai_summary_page()
    elif st.session_state.current_page == "Statistics":
        statistics_page()
    elif st.session_state.current_page == "View Entries":
        view_entries_page()
    elif st.session_state.current_page == "Download PDF":
        download_pdf_page()
    elif st.session_state.current_page == "Mental Support":
        mental_support_page()
    elif st.session_state.current_page == "FAQ":
        faq_page()
    elif st.session_state.current_page == "About":
        about_page()

def add_entry_page():
    """Page for adding new diary entries."""
    
    # Check if high risk was detected and show persistent redirect buttons
    if st.session_state.get('high_risk_detected', False):
        st.markdown("""
        <div class="feature-card" style="background: #ffe6e6; border: 3px solid #ff4444; margin-bottom: 2rem;">
            <h2 style="color: #ff4444; text-align: center;">🚨 URGENT: High Risk Detected</h2>
            <p style="text-align: center; font-size: 1.2em;">We're here to help you. Please redirect to mental health support.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if emergency contacts exist
        import json
        import os
        emergency_contacts = []
        contacts_file = st.session_state.app.config.EMERGENCY_CONTACTS_JSON
        
        if os.path.exists(contacts_file):
            try:
                with open(contacts_file, 'r') as f:
                    emergency_contacts = json.load(f)
            except Exception as e:
                print(f"❌ Error loading emergency contacts: {e}")
        
        # Show warning if no contacts exist (but don't show form)
        if not emergency_contacts:
            st.warning("⚠️ No emergency contacts configured. WhatsApp alerts cannot be sent. Please go to Mental Support page to add contacts.")
        
        # Redirect buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏃 Go to Mental Support NOW", type="primary", use_container_width=True, key="persistent_redirect_btn"):
                # Send WhatsApp alerts if enabled and contacts exist
                if st.session_state.app.config.AUTO_ALERT_ENABLED and emergency_contacts:
                    # Check if alerts have already been sent to prevent duplicates
                    if not st.session_state.get('whatsapp_alerts_sent', False):
                        try:
                            alert_message = f"🚨 CRISIS ALERT: High suicide risk detected in MindLens diary entry. Please check on this person immediately."
                            
                            if hasattr(st.session_state.app.suicide_detector, 'send_whatsapp_alert'):
                                # Use the send_whatsapp_alert method which handles deduplication
                                success = st.session_state.app.suicide_detector.send_whatsapp_alert(
                                    st.session_state.app.config.EMERGENCY_CONTACTS_JSON,
                                    alert_message
                                )
                            else:
                                from suicide_detector_fallback import SuicideDetectorFallback
                                detector = SuicideDetectorFallback("", "")
                                # Use the send_whatsapp_alert method which handles deduplication
                                success = detector.send_whatsapp_alert(
                                    st.session_state.app.config.EMERGENCY_CONTACTS_JSON,
                                    alert_message
                                )
                            
                            if success:
                                # Set flag to prevent duplicate sending
                                st.session_state.whatsapp_alerts_sent = True
                                print("✅ WhatsApp alerts sent for persistent high-risk alert")
                            else:
                                print("❌ Failed to send WhatsApp alerts")
                        except Exception as e:
                            print(f"❌ Error sending alerts: {e}")
                                
                # Clear the high risk flag and redirect
                print("🔄 Redirecting to Mental Support page...")
                st.session_state.high_risk_detected = False
                st.session_state.current_page = "Mental Support"
                st.rerun()
            
            if st.button("⏸️ I'm Safe - Continue", use_container_width=True, key="persistent_cancel_btn"):
                st.session_state.high_risk_detected = False
                st.success("✅ Okay. Please reach out if you need support.")
                st.rerun()
        
        st.markdown("---")
    
    # Modern hero section with gradient
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
            📝 Create Your Memory
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin-top: 1rem; font-size: 1.2rem; font-weight: 300;">
            Capture your thoughts, feelings, and moments. Let AI help you understand your emotional journey.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date selection in a beautiful card
    st.markdown("""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    ">
        <h3 style="margin: 0 0 1rem 0; color: #333; font-size: 1.3rem;">📅 When did this happen?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        entry_date = st.date_input(
            "📅 Date",
            value=date.today(),
            help="Select the date for this memory",
            label_visibility="collapsed"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # If a transcription was just created, set the widget state BEFORE rendering the widget
    if st.session_state.get("_fill_entry_text_once"):
        st.session_state["entry_text"] = st.session_state.get("_entry_text_buffer", "")
        st.session_state.pop("_fill_entry_text_once", None)

    # Text input in a beautiful card
    st.markdown("""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    ">
        <h3 style="margin: 0 0 1rem 0; color: #333; font-size: 1.3rem;">💭 Share Your Thoughts</h3>
        <p style="color: #666; margin: 0; font-size: 0.95rem;">Express yourself freely. What’s on your mind today?</p>
    </div>
    """, unsafe_allow_html=True)
    
    entry_text = st.text_area(
        "Your thoughts",
        height=220,
        placeholder="✨ Today I felt... \n\n🌟 What happened: \n\n💖 How I’m feeling: \n\n🎯 What I’m grateful for: ",
        label_visibility="collapsed",
        key="entry_text",
        help="Write freely about your day, feelings, experiences, or anything you'd like to remember"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Multimedia Upload Section Header
    st.markdown("""
    <div style="
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    ">
        <h2 style="
            color: #333;
            font-size: 2rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
        ">🎨 Enhance Your Memory</h2>
        <p style="
            color: #666;
            font-size: 1.1rem;
            margin: 0;
        ">Add audio, video, or images to capture the complete experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout for Audio and Video
    col_audio, col_video = st.columns(2, gap="large")
    
    with col_audio:
        # Audio Transcription Card
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.8rem 1.5rem;
            border-radius: 18px;
            box-shadow: 0 8px 25px rgba(240, 147, 251, 0.35);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎤</div>
                <h3 style="
                    color: white;
                    margin: 0 0 0.3rem 0;
                    font-size: 1.3rem;
                    font-weight: 700;
                    text-shadow: 2px 2px 6px rgba(0,0,0,0.5), 0 0 10px rgba(0,0,0,0.3);
                ">Audio Transcription</h3>
                <p style="
                    color: white;
                    margin: 0;
                    font-size: 0.95rem;
                    font-weight: 500;
                    text-shadow: 1px 1px 4px rgba(0,0,0,0.4), 0 0 8px rgba(0,0,0,0.2);
                ">Convert speech to text</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            help="Supported formats: WAV, MP3, M4A",
            key="audio_uploader"
        )
    if audio_file is not None:
        if st.button("📝 Transcribe Audio", use_container_width=True):
            try:
                from faster_whisper import WhisperModel
                # Ensure model dir exists
                Path(WHISPER_MODEL_DIR).mkdir(parents=True, exist_ok=True)
                # Load local model
                model = WhisperModel(model_size_or_path=str(WHISPER_MODEL_DIR), device="cpu")
                # Save temp audio
                tmp_audio_path = Path("data") / "tmp_audio_input"
                tmp_audio_path.mkdir(parents=True, exist_ok=True)
                audio_path = tmp_audio_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.name}"
                with open(audio_path, "wb") as f:
                    f.write(audio_file.read())
                segments, info = model.transcribe(str(audio_path), beam_size=5)
                transcript = "".join([seg.text for seg in segments]).strip()
                if transcript:
                    # Save into a buffer and set a one-shot flag so we can write to the widget state before render
                    st.session_state["_entry_text_buffer"] = transcript
                    st.session_state["_fill_entry_text_once"] = True
                    st.success("✅ Transcription complete. Inserted into text area.")
                    st.rerun()
                else:
                    st.info("No speech detected.")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

    # Keep local variable in sync (optional)
    entry_text = st.session_state.get("entry_text", entry_text)
    
    with col_video:
        # Video Upload Card
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.8rem 1.5rem;
            border-radius: 18px;
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.35);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎥</div>
                <h3 style="
                    color: white;
                    margin: 0 0 0.3rem 0;
                    font-size: 1.3rem;
                    font-weight: 700;
                    text-shadow: 2px 2px 6px rgba(0,0,0,0.5), 0 0 10px rgba(0,0,0,0.3);
                ">Video Upload</h3>
                <p style="
                    color: white;
                    margin: 0;
                    font-size: 0.95rem;
                    font-weight: 500;
                    text-shadow: 1px 1px 4px rgba(0,0,0,0.4), 0 0 8px rgba(0,0,0,0.2);
                ">Transcribe or attach video</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        video_file = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
            key="video_uploader",
            label_visibility="collapsed",
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
    
    if video_file is not None:
        # Option to transcribe or just attach video
        transcribe_video = st.checkbox(
            "🎤 Transcribe video audio to text",
            value=False,
            help="Check this to extract and transcribe audio from video. Leave unchecked to just upload the video with your text.",
            key="transcribe_video_checkbox"
        )
        
        if transcribe_video:
            # Show transcribe button
            if st.button("🎥 Transcribe Video to Text", use_container_width=True, key="transcribe_video_btn", type="primary"):
                try:
                    import subprocess
                    from faster_whisper import WhisperModel
                    
                    # Create temp directories
                    tmp_video_path = Path("data") / "tmp_video_input"
                    tmp_video_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save uploaded video
                    video_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.name}"
                    video_path = tmp_video_path / video_filename
                    
                    with st.spinner("💾 Saving video file..."):
                        with open(video_path, "wb") as f:
                            f.write(video_file.read())
                    
                    # Extract audio from video using FFmpeg
                    audio_filename = f"{video_filename.rsplit('.', 1)[0]}.wav"
                    audio_path = tmp_video_path / audio_filename
                    
                    with st.spinner("🎥 Extracting audio from video..."):
                        # FFmpeg command to extract audio
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-i", str(video_path),
                            "-vn",  # No video
                            "-acodec", "pcm_s16le",  # Audio codec
                            "-ar", "16000",  # Sample rate 16kHz (good for Whisper)
                            "-ac", "1",  # Mono audio
                            "-y",  # Overwrite output file
                            str(audio_path)
                        ]
                        
                        result = subprocess.run(
                            ffmpeg_cmd,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minutes timeout
                        )
                        
                        if result.returncode != 0:
                            st.error(f"❌ FFmpeg error: {result.stderr}")
                            raise Exception(f"FFmpeg failed with return code {result.returncode}")
                        
                        st.success("✅ Audio extracted successfully!")
                    
                    # Transcribe the extracted audio
                    with st.spinner("🎤 Transcribing audio to text..."):
                        # Ensure model dir exists
                        Path(WHISPER_MODEL_DIR).mkdir(parents=True, exist_ok=True)
                        # Load local Whisper model
                        model = WhisperModel(model_size_or_path=str(WHISPER_MODEL_DIR), device="cpu")
                        segments, info = model.transcribe(str(audio_path), beam_size=5)
                        transcript = "".join([seg.text for seg in segments]).strip()
                    
                    # Clean up temporary files
                    try:
                        video_path.unlink()
                        audio_path.unlink()
                    except:
                        pass
                    
                    if transcript:
                        # Save into a buffer and set a one-shot flag
                        st.session_state["_entry_text_buffer"] = transcript
                        st.session_state["_fill_entry_text_once"] = True
                        st.success("✅ Video transcription complete! Text inserted into 'Your Thoughts' area.")
                        st.info(f"📝 Transcribed {len(transcript)} characters from video.")
                        st.rerun()
                    else:
                        st.warning("⚠️ No speech detected in video.")
                        
                except subprocess.TimeoutExpired:
                    st.error("❌ Transcription timed out. Please try a shorter video (max 5 minutes).")
                except FileNotFoundError:
                    st.error("❌ FFmpeg is not installed on your system.")
                    st.markdown("""
                    <div class="feature-card" style="background: #fff3cd; border-left: 6px solid #ffc107;">
                        <h4>💡 How to install FFmpeg:</h4>
                        <ol>
                            <li><strong>Windows:</strong> Download from <a href="https://ffmpeg.org/download.html" target="_blank">ffmpeg.org</a> and add to PATH</li>
                            <li><strong>Mac:</strong> Run <code>brew install ffmpeg</code></li>
                            <li><strong>Linux:</strong> Run <code>sudo apt-get install ffmpeg</code></li>
                        </ol>
                        <p>After installation, restart the application.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Video transcription failed: {e}")
                    with st.expander("🔍 View error details"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            # User doesn't want to transcribe - just inform them
            st.info("📝 Video uploaded. Write your thoughts in the text area above and click 'Add Entry' to save.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Image upload section - Enhanced full-width card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(250, 112, 154, 0.4);
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.8rem;">🖼️</div>
        <h3 style="
            color: white;
            margin: 0 0 0.5rem 0;
            font-size: 1.8rem;
            font-weight: 700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5), 0 0 15px rgba(0,0,0,0.3);
        ">Add a Picture to Your Memory</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.1rem;
            font-weight: 500;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.4), 0 0 10px rgba(0,0,0,0.2);
        ">A picture is worth a thousand words - attach a photo to make this moment unforgettable</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_img1, col_img2 = st.columns([1, 1], gap="medium")
    
    with col_img1:
        img_file = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            help="Supported formats: PNG, JPG, JPEG"
        )
    
    with col_img2:
        image_desc = st.text_input(
            "📝 Image description",
            placeholder="What's in this photo? (beach sunset, birthday cake, mountain view...)",
            label_visibility="collapsed",
            help="Add a description to make your image searchable later"
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Submit button section with enhanced styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">✨</div>
        <h3 style="
            color: white;
            margin: 0 0 0.5rem 0;
            font-size: 1.6rem;
            font-weight: 700;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5), 0 0 15px rgba(0,0,0,0.3);
        ">Ready to Preserve This Moment?</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.05rem;
            font-weight: 500;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.4), 0 0 10px rgba(0,0,0,0.2);
        ">Your memory will be analyzed with AI to understand emotions and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Submit button - larger and more prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Save My Memory ✨", type="primary", use_container_width=True):
            if entry_text.strip():
                with st.spinner("🤖 Processing your entry..."):
                    try:
                        saved_image_path = None
                        if img_file is not None:
                            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                            img_save = IMAGES_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{img_file.name}"
                            with open(img_save, "wb") as f:
                                f.write(img_file.read())
                            saved_image_path = str(img_save.relative_to(Path.cwd()))

                        # Save video file if uploaded
                        saved_video_path = None
                        if video_file is not None:
                            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
                            video_save = VIDEOS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.name}"
                            with open(video_save, "wb") as f:
                                f.write(video_file.getvalue())  # getvalue() to read uploaded file
                            saved_video_path = str(video_save.relative_to(Path.cwd()))

                        result = st.session_state.app.process_entry(
                            text=entry_text,
                            date=entry_date.isoformat(),
                            image_path=saved_image_path,
                            image_desc=image_desc.strip() if image_desc else None,
                            video_path=saved_video_path
                        )
                        
                        st.markdown("""
                        <div class="success-message">
                            <h3>🎉 Entry Added Successfully!</h3>
                            <p>Your thoughts have been processed and stored with AI insights.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display results in cards
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>😊 Sentiment</h3>
                                <h2>{result['sentiment'].title()}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🎭 Emotions</h3>
                                <h2>{len(result['emotions'])}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🏷️ Tags</h3>
                                <h2>{len(result['tags'])}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            # Suicide risk assessment
                            suicide_score = result.get('suicide_score')
                            if suicide_score is not None:
                                if suicide_score >= 0.5:
                                    risk_level = "High Risk"
                                    risk_color = "#ff6b6b"
                                    icon = "🔴"
                                else:
                                    risk_level = "Low Risk"
                                    risk_color = "#4caf50"
                                    icon = "🟢"
                                
                                st.markdown(f"""
                                <div class="metric-card" style="border-left: 6px solid {risk_color};">
                                    <h3>🛡️ Mental Health</h3>
                                    <h2 style="color: {risk_color};">{icon} {risk_level}</h2>
                                    <p style="font-size: 0.8em; margin: 0;">Score: {suicide_score:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🛡️ Mental Health</h3>
                                    <h2>N/A</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show detected emotions and tags
                        if result['emotions']:
                            st.markdown("### 🎭 Detected Emotions")
                            emotion_html = ""
                            for emotion in result['emotions']:
                                emotion_html += f'<span class="emotion-badge">{emotion}</span>'
                            st.markdown(emotion_html, unsafe_allow_html=True)
                        
                        if result['tags']:
                            st.markdown("### 🏷️ Context Tags")
                            tag_html = ""
                            for tag in result['tags']:
                                tag_html += f'<span class="tag-badge">{tag}</span>'
                            st.markdown(tag_html, unsafe_allow_html=True)
                        
                        # Show suicide risk analysis if available
                        if result.get('suicide_score') is not None:
                            st.markdown("### 🛡️ Mental Health Analysis")
                            suicide_score = result.get('suicide_score', 0.0)
                            suicide_prediction = result.get('suicide_prediction', 'Unknown')
                            suicide_confidence = result.get('suicide_confidence', 0.0)
                            suicide_emotion = result.get('suicide_emotion', 'Unknown')
                            suicide_tags = result.get('suicide_tags', [])
                            
                            # Determine colors and styling
                            if suicide_score >= 0.5:
                                card_color = "#ff6b6b"
                                bg_color = "#ffe6e6"
                                icon = "🔴"
                                risk_text = "High Risk"
                            else:
                                card_color = "#4caf50"
                                bg_color = "#e8f5e8"
                                icon = "🟢"
                                risk_text = "Low Risk"
                            
                            # Main risk assessment card
                            st.markdown(f"""
                            <div style="background: {bg_color}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 6px solid {card_color};">
                                <h4 style="color: {card_color}; margin: 0 0 0.5rem 0;">{icon} {risk_text}</h4>
                                <p style="margin: 0; font-size: 1.1em;"><strong>Confidence:</strong> {suicide_confidence:.1f}% | <strong>Risk Score:</strong> {suicide_score:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card" style="border-left: 6px solid {card_color};">
                                    <h4>😊 Detected Emotion</h4>
                                    <p style="color: {card_color}; font-size: 1.2em; margin: 0;">{suicide_emotion}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                if suicide_tags:
                                    tag_html = ""
                                    for tag in suicide_tags[:5]:  # Show first 5 tags
                                        tag_html += f'<span class="tag" style="background: {card_color}20; color: {card_color}; border: 1px solid {card_color}; padding: 0.2rem 0.5rem; border-radius: 15px; margin: 0.2rem; display: inline-block; font-size: 0.8em;">{tag.replace("_", " ").title()}</span>'
                                    
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>🏷️ Key Indicators</h4>
                                        <div style="margin-top: 0.5rem;">{tag_html}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Motivational quote
                            if st.session_state.app.suicide_detector.is_loaded():
                                quote = st.session_state.app.suicide_detector.get_motivational_quote(suicide_score)
                                
                                if suicide_score >= 0.5:
                                    st.markdown(f"""
                                    <div class="delete-warning">
                                        <h4>💪 You Are Stronger Than You Know</h4>
                                        <p style="font-style: italic; margin: 1rem 0;">"{quote}"</p>
                                        <p><strong>Remember:</strong> You don't have to face this alone. Consider reaching out to a trusted friend, family member, or mental health professional.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="delete-info">
                                        <h4>🌟 Keep Shining!</h4>
                                        <p style="font-style: italic; margin: 1rem 0;">"{quote}"</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Crisis resources for high risk
                            if suicide_score >= 0.5:
                                # Store high risk detection in session state
                                st.session_state.high_risk_detected = True
                                st.session_state.last_entry_date = result.get('date')
                                
                                # Send WhatsApp alerts automatically if enabled
                                if st.session_state.app.config.AUTO_ALERT_ENABLED:
                                    # Check if alerts have already been sent to prevent duplicates
                                    if not st.session_state.get('whatsapp_alerts_sent', False):
                                        try:
                                            alert_message = f"🚨 CRISIS ALERT: High suicide risk detected in MindLens diary entry. Please check on this person immediately."
                                            
                                            if hasattr(st.session_state.app.suicide_detector, 'send_whatsapp_alert'):
                                                # Use the send_whatsapp_alert method which handles deduplication
                                                success = st.session_state.app.suicide_detector.send_whatsapp_alert(
                                                    st.session_state.app.config.EMERGENCY_CONTACTS_JSON,
                                                    alert_message
                                                )
                                            else:
                                                from suicide_detector_fallback import SuicideDetectorFallback
                                                detector = SuicideDetectorFallback("", "")
                                                # Use the send_whatsapp_alert method which handles deduplication
                                                success = detector.send_whatsapp_alert(
                                                    st.session_state.app.config.EMERGENCY_CONTACTS_JSON,
                                                    alert_message
                                                )
                                            
                                            if success:
                                                # Set flag to prevent duplicate sending
                                                st.session_state.whatsapp_alerts_sent = True
                                                print("✅ WhatsApp alerts sent automatically for high-risk content")
                                            else:
                                                print("❌ Failed to send WhatsApp alerts")
                                        except Exception as e:
                                            print(f"❌ Error sending alerts: {e}")
                                
                                st.markdown("""
                                <div class="feature-card" style="background: #fff3cd; border: 1px solid #ffeaa7;">
                                    <h4>🆘 Crisis Resources</h4>
                                    <p><strong>If you're having thoughts of suicide:</strong></p>
                                    <ul>
                                        <li><strong>National Suicide Prevention Lifeline:</strong> 988 (24/7)</li>
                                        <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
                                        <li><strong>Emergency:</strong> Call 911</li>
                                    </ul>
                                    <p>Remember: You are not alone, and help is available.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show simple redirect message
                                st.markdown("### 🛡️ High Risk Detected - Crisis Support")
                                st.markdown("🚨 **URGENT:** We're connecting you to mental health professionals and support services.")
                                
                                # Simple redirect message without countdown
                                st.markdown(
                                    "<div style='text-align: center; font-size: 1.8em; margin: 30px 0; font-family: sans-serif; color: #ff4444; background: #ffe6e6; padding: 25px; border-radius: 10px; border: 3px solid #ff4444;'><strong>🔄 You will be redirected to the Mental Support page</strong></div>",
                                    unsafe_allow_html=True
                                )
                                
                                # Buttons for user action
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("🏃 Go to Mental Support Now", type="primary", use_container_width=True, key="btn_redirect_mental_support"):
                                        # Set redirect flag and trigger redirect
                                        print("🔄 Redirecting to Mental Support page...")
                                        st.session_state.high_risk_detected = False
                                        # Clear the alert sent flag for future alerts
                                        if 'whatsapp_alerts_sent' in st.session_state:
                                            del st.session_state.whatsapp_alerts_sent
                                        st.session_state.current_page = "Mental Support"
                                        st.rerun()
                                
                                with col2:
                                    if st.button("⏸️ I'm Safe - No Redirect Needed", use_container_width=True, key="btn_cancel_redirect"):
                                        st.session_state.high_risk_detected = False
                                        st.success("✅ Understood. Please reach out if you need support.")

                        # Show image if provided
                        if result.get('image_path'):
                            st.markdown("### 🖼️ Image Attached")
                            try:
                                st.image(str(result['image_path']))
                                if result.get('image_desc'):
                                    st.caption(f"Description: {result['image_desc']}")
                            except Exception:
                                st.info("Image saved but could not be previewed.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing entry: {e}")
            else:
                st.warning("⚠️ Please enter some text for your entry.")

def search_entries_page():
    """Page for searching diary entries."""
    st.markdown("""
    <div class="feature-card">
        <h2>🔍 Search Entries</h2>
        <p>Find your past thoughts using natural language. Search by emotions, topics, or any keywords.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search input
    st.markdown("### 🔎 What are you looking for?")
    search_query = st.text_input(
        "Search query",
        placeholder="e.g., 'happy moments', 'work stress', 'family time', 'when I felt proud'",
        label_visibility="collapsed"
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎭 Filter by Emotions")
        filter_emotions = st.multiselect(
            "Emotions",
            options=["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "pride", "excitement", "anxiety"],
            help="Select emotions to filter by",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### 🏷️ Filter by Tags")
        filter_tags = st.multiselect(
            "Tags",
            options=["work", "family", "health", "travel", "study_learning", "relationships", "leisure_entertainment", "routine_chores"],
            help="Select context tags to filter by",
            label_visibility="collapsed"
        )
    
    # Number of results
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        top_k = st.slider("📊 Number of results", 1, 20, 5)
    
    # Image description retrieval
    st.markdown("#### 🖼️ Retrieve by Image Description")
    image_desc_query = st.text_input("Image description label", placeholder="Type the image description used during upload…")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🖼️ Retrieve Image", use_container_width=True):
            if image_desc_query.strip():
                with st.spinner("🖼️ Retrieving images by description…"):
                    try:
                        img_results = st.session_state.app.search_entries(
                            query=image_desc_query,
                            top_k=top_k,
                            filter_emotions=filter_emotions if filter_emotions else None,
                            filter_tags=filter_tags if filter_tags else None
                        )
                        # Keep only entries that have images and strongly prefer description match
                        desc_lower = image_desc_query.lower()
                        filtered = []
                        for r in img_results:
                            if not r.get('image_path'):
                                continue
                            img_desc = (r.get('image_desc') or '').lower()
                            txt = (r.get('text') or '').lower()
                            if desc_lower and (desc_lower in img_desc or desc_lower in txt):
                                filtered.append(r)
                        img_results = filtered
                        if img_results:
                            st.markdown(f"""
                            <div class="success-message">
                                <h3>🖼️ Found {len(img_results)} image entries!</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            for i, result in enumerate(img_results, 1):
                                with st.expander(f"🖼️ Image Entry {i} - {result.get('date')}", expanded=True):
                                    if result.get('image_path'):
                                        try:
                                            st.image(str(result.get('image_path')))
                                        except Exception:
                                            st.info("Image associated but could not be previewed.")
                                    if result.get('image_desc'):
                                        st.caption(f"Description: {result.get('image_desc')}")
                                    
                                    # Show video if available
                                    if result.get('video_path'):
                                        st.markdown("### 🎥 Video")
                                        try:
                                            video_path = Path(result.get('video_path'))
                                            if video_path.exists():
                                                st.video(str(video_path))
                                            else:
                                                st.warning(f"⚠️ Video file not found: {result.get('video_path')}")
                                        except Exception as e:
                                            st.info(f"Video associated but could not be previewed: {e}")
                                    
                                    st.markdown(f"""
                                    <div class=\"search-result\"> 
                                        <p><strong>Text:</strong> {result.get('text')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error retrieving images: {e}")
            else:
                st.warning("⚠️ Please enter an image description label.")

    # Search button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if search_query.strip():
                with st.spinner("🔍 Searching your memories..."):
                    try:
                        results = st.session_state.app.search_entries(
                            query=search_query,
                            top_k=top_k,
                            filter_emotions=filter_emotions if filter_emotions else None,
                            filter_tags=filter_tags if filter_tags else None
                        )
                        
                        if results:
                            st.markdown(f"""
                            <div class="success-message">
                                <h3>🎯 Found {len(results)} matching entries!</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, result in enumerate(results, 1):
                                with st.expander(f"📝 Entry {i} - {result.get('date')}", expanded=True):
                                    st.markdown(f"""
                                    <div class="search-result">
                                        <p><strong>Text:</strong> {result.get('text')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show metadata in cards
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h3>😊 Sentiment</h3>
                                            <h2>{result.get('sentiment').title()}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        emotions = result.get('emotions', [])
                                        emotion_text = ', '.join(emotions) if emotions else 'None'
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h3>🎭 Emotions</h3>
                                            <h2>{emotion_text}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col3:
                                        tags = result.get('tags', [])
                                        tag_text = ', '.join(tags) if tags else 'None'
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <h3>🏷️ Tags</h3>
                                            <h2>{tag_text}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Show image preview if available
                                    if result.get('image_path'):
                                        st.markdown("### 🖼️ Image")
                                        try:
                                            st.image(str(result.get('image_path')))
                                            if result.get('image_desc'):
                                                st.caption(f"Description: {result.get('image_desc')}")
                                        except Exception:
                                            st.info("Image associated but could not be previewed.")
                                    
                                    # Show video if available
                                    if result.get('video_path'):
                                        st.markdown("### 🎥 Video")
                                        try:
                                            video_path = Path(result.get('video_path'))
                                            if video_path.exists():
                                                st.video(str(video_path))
                                            else:
                                                st.warning(f"⚠️ Video file not found: {result.get('video_path')}")
                                        except Exception as e:
                                            st.info(f"Video associated but could not be previewed: {e}")

                                    # Show suicide risk if available
                                    if result.get('suicide_score') is not None:
                                        suicide_score = result.get('suicide_score', 0.0)
                                        suicide_prediction = result.get('suicide_prediction', 'Unknown')
                                        if suicide_score >= 0.5:
                                            risk_color = "#ff6b6b"
                                            risk_level = "High Risk"
                                        else:
                                            risk_color = "#4caf50"
                                            risk_level = "Low Risk"
                                        
                                        st.markdown(f"""
                                        <div class="metric-card" style="border-left: 6px solid {risk_color}; margin: 1rem 0;">
                                            <h3>🛡️ Mental Health Risk</h3>
                                            <h2 style="color: {risk_color};">{risk_level}</h2>
                                            <p style="font-size: 0.8em; margin: 0;">Score: {suicide_score:.2f}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Show similarity score if available
                                    if 'distance' in result:
                                        similarity = 1 - result['distance']  # Convert distance to similarity
                                        st.metric("🎯 Similarity Score", f"{similarity:.2f}")
                        else:
                            st.info("🔍 No matching entries found. Try a different search query or adjust filters.")
                            
                    except Exception as e:
                        st.error(f"❌ Error searching entries: {e}")
            else:
                st.warning("⚠️ Please enter a search query.")

def ai_summary_page():
    """Page for AI-powered summaries."""
    st.markdown("""
    <div class="feature-card">
        <h2>🤖 AI Summary</h2>
        <p>Get intelligent insights about patterns in your thoughts and feelings using AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💭 What would you like to know about?")
    summary_query = st.text_input(
        "Summary query",
        placeholder="e.g., 'How have I been feeling about work lately?', 'What makes me happy?', 'My stress patterns', 'When do I feel most productive?'",
        label_visibility="collapsed"
    )
    
    # Number of entries to include
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        top_k = st.slider("📊 Number of entries to analyze", 3, 15, 5)
    
    # Generate summary button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            if summary_query.strip():
                with st.spinner("🤖 Analyzing your thoughts with AI..."):
                    try:
                        summary = st.session_state.app.get_ai_summary(
                            query=summary_query,
                            top_k=top_k
                        )
                        
                        st.markdown("""
                        <div class="success-message">
                            <h3>🎉 AI Summary Generated!</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="summary-box">
                            <h3>🧠 AI Insights</h3>
                            <p>{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show related entries (text + image) for the same query
                        try:
                            related = st.session_state.app.search_entries(
                                query=summary_query,
                                top_k=top_k
                            )
                            if related:
                                st.markdown("### 🔗 Related Entries")
                                for i, result in enumerate(related, 1):
                                    with st.expander(f"📝 Entry {i} - {result.get('date')}", expanded=False):
                                        st.markdown(f"""
                                        <div class=\"search-result\">
                                            <p><strong>Text:</strong> {result.get('text')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        if result.get('image_path'):
                                            st.markdown("#### 🖼️ Image")
                                            try:
                                                st.image(str(result.get('image_path')))
                                                if result.get('image_desc'):
                                                    st.caption(f"Description: {result.get('image_desc')}")
                                            except Exception:
                                                st.info("Image associated but could not be previewed.")
                                        if 'distance' in result:
                                            similarity = 1 - result['distance']
                                            st.metric("🎯 Similarity", f"{similarity:.2f}")
                        except Exception as e:
                            st.info(f"Could not load related entries: {e}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {e}")
            else:
                st.warning("⚠️ Please enter a query for the summary.")

def statistics_page():
    """Page for viewing statistics."""
    st.markdown("""
    <div class="feature-card">
        <h2>📊 Statistics</h2>
        <p>View system information and model status for your MindLens application.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        stats = st.session_state.app.get_stats()
        
        st.markdown("### 🖥️ System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🗄️ Vector Store</h3>
                <h2>{stats['vector_store'].title()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 LLM Backend</h3>
                <h2>{stats['llm_backend'].title()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🧠 Model Status")
        models = stats['models_loaded']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Loaded" if models['emotion_model'] else "❌ Not loaded"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎭 Emotion Model</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅ Loaded" if models['context_tagger'] else "❌ Not loaded"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Context Tagger</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status = "✅ Loaded" if models['embedder'] else "❌ Not loaded"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 Embedder</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Add comprehensive analytics dashboard
        st.markdown("### 📊 Detailed Analytics Dashboard")
        
        try:
            # Get real data from entries
            entries_file = Path("data/entries.jsonl")
            if entries_file.exists():
                entries = []
                with open(entries_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
                
                if entries:
                    # Process real data
                    df = pd.DataFrame(entries)
                    df['date'] = pd.to_datetime(df['date'])
                    df['text_length'] = df['text'].str.len()
                    df['word_count'] = df['text'].str.split().str.len()
                    
                    # Extract hour from timestamp if available
                    if 'timestamp' in df.columns:
                        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    else:
                        df['hour'] = 12  # Default to noon if no timestamp
                    
                    df['day_of_week'] = df['date'].dt.day_name()
                    df['week'] = df['date'].dt.to_period('W')
                    
                    # Calculate sentiment score (1=positive, 0=neutral, -1=negative)
                    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)
                    
                    # === 7) KPIs (Top of Dashboard) ===
                    st.markdown("### 🎯 Key Performance Indicators")
                    
                    # Calculate KPIs
                    last_7_days = df[df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
                    last_30_days = df[df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
                    
                    avg_sentiment_7d = last_7_days['sentiment_score'].mean() if len(last_7_days) > 0 else 0
                    
                    # Check for suicidal thoughts
                    suicide_flagged = df['suicide_score'] >= 0.5 if 'suicide_score' in df.columns else pd.Series([False] * len(df))
                    suicide_pct_30d = (suicide_flagged[df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))].sum() / len(last_30_days) * 100) if len(last_30_days) > 0 else 0
                    
                    # Calculate longest positive streak
                    positive_streak = 0
                    max_positive_streak = 0
                    for score in df.sort_values('date')['sentiment_score']:
                        if score > 0:
                            positive_streak += 1
                            max_positive_streak = max(max_positive_streak, positive_streak)
                        else:
                            positive_streak = 0
                    
                    # Most frequent emotion and context
                    all_emotions = []
                    for emotions_list in last_30_days['emotions']:
                        if emotions_list:
                            all_emotions.extend(emotions_list)
                    most_freq_emotion = pd.Series(all_emotions).value_counts().index[0] if all_emotions else 'N/A'
                    
                    all_tags = []
                    for tags_list in last_30_days['tags']:
                        if tags_list:
                            all_tags.extend(tags_list)
                    most_freq_context = pd.Series(all_tags).value_counts().index[0] if all_tags else 'N/A'
                    
                    # Display KPIs with improved formatting
                    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                    
                    with kpi1:
                        # Sentiment: -1 (negative) to +1 (positive)
                        sentiment_emoji = "😊" if avg_sentiment_7d > 0.3 else "😐" if avg_sentiment_7d >= -0.3 else "😔"
                        sentiment_label = "Positive" if avg_sentiment_7d > 0.3 else "Neutral" if avg_sentiment_7d >= -0.3 else "Negative"
                        
                        st.metric(
                            "7-Day Avg Mood",
                            f"{sentiment_emoji} {sentiment_label}",
                            delta=f"Score: {avg_sentiment_7d:+.2f}" if len(last_7_days) > 0 else "No data",
                            help="Average sentiment from -1 (very negative) to +1 (very positive)"
                        )
                    
                    with kpi2:
                        # Risk percentage with better display
                        risk_count = int(suicide_flagged[df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))].sum())
                        risk_color = "🟢" if suicide_pct_30d == 0 else "🟡" if suicide_pct_30d < 10 else "🔴"
                        
                        st.metric(
                            "Risk Alerts (30d)",
                            f"{risk_color} {risk_count} entries" if len(last_30_days) > 0 else "🟢 No data",
                            delta=f"{suicide_pct_30d:.1f}% of total" if len(last_30_days) > 0 else None,
                            delta_color="inverse" if suicide_pct_30d > 0 else "off",
                            help=f"High-risk entries detected in last 30 days ({risk_count} out of {len(last_30_days)} total entries)"
                        )
                    
                    with kpi3:
                        streak_display = f"🔥 {max_positive_streak}" if max_positive_streak > 0 else "—"
                        streak_label = "days" if max_positive_streak != 1 else "day"
                        
                        st.metric(
                            "Longest Positive Streak",
                            f"{streak_display} {streak_label}" if max_positive_streak > 0 else "No streak yet",
                            help="Consecutive days with positive sentiment (score > 0)"
                        )
                    
                    with kpi4:
                        st.metric(
                            "Top Emotion (30d)",
                            f"🎭 {most_freq_emotion}",
                            help="Most frequently detected emotion in last 30 days"
                        )
                    
                    with kpi5:
                        st.metric(
                            "Top Context (30d)",
                            f"🏷️ {most_freq_context}",
                            help="Most common topic/context in last 30 days"
                        )
                    
                    st.markdown("---")
                    
                    # === 1) Trends Over Time ===
                    st.markdown("### 📈 Trends Over Time")
                    
                    # Daily sentiment with 7-day rolling average
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Daily Sentiment & 7-Day Average")
                        daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
                        daily_sentiment['rolling_avg'] = daily_sentiment['sentiment_score'].rolling(window=7, min_periods=1).mean()
                        
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily_sentiment['date'],
                            y=daily_sentiment['sentiment_score'],
                            mode='lines+markers',
                            name='Daily Sentiment',
                            line=dict(color='lightblue', width=1),
                            marker=dict(size=4)
                        ))
                        fig.add_trace(go.Scatter(
                            x=daily_sentiment['date'],
                            y=daily_sentiment['rolling_avg'],
                            mode='lines',
                            name='7-Day Average',
                            line=dict(color='blue', width=3)
                        ))
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date", yaxis_title="Sentiment")
                        st.plotly_chart(fig, use_container_width=True, key="daily_sentiment_chart")
                    
                    with col2:
                        st.markdown("#### Weekly Average Sentiment")
                        weekly_sentiment = df.groupby('week')['sentiment_score'].mean().reset_index()
                        weekly_sentiment['week_str'] = weekly_sentiment['week'].astype(str)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=weekly_sentiment['week_str'],
                                y=weekly_sentiment['sentiment_score'],
                                marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in weekly_sentiment['sentiment_score']]
                            )
                        ])
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Week", yaxis_title="Avg Sentiment")
                        st.plotly_chart(fig, use_container_width=True, key="weekly_sentiment_chart")
                    
                    # Stacked area of emotions over time
                    st.markdown("#### Emotional Mix Over Time")
                    emotion_data = []
                    for idx, row in df.iterrows():
                        for emotion in row['emotions'] if row['emotions'] else []:
                            emotion_data.append({'date': row['date'], 'emotion': emotion})
                    
                    if emotion_data:
                        emotion_df = pd.DataFrame(emotion_data)
                        emotion_pivot = emotion_df.groupby(['date', 'emotion']).size().unstack(fill_value=0)
                        
                        fig = go.Figure()
                        for emotion in emotion_pivot.columns:
                            fig.add_trace(go.Scatter(
                                x=emotion_pivot.index,
                                y=emotion_pivot[emotion],
                                mode='lines',
                                name=emotion,
                                stackgroup='one'
                            ))
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date", yaxis_title="Emotion Count")
                        st.plotly_chart(fig, use_container_width=True, key="emotion_stacked_area_chart")
                    
                    st.markdown("---")
                    
                    # === 2) Risk & Safety Signals ===
                    st.markdown("### 🚨 Risk & Safety Signals")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Timeline of Flagged Entries")
                        if 'suicide_score' in df.columns:
                            flagged_df = df[df['suicide_score'] >= 0.5].copy()
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=flagged_df['date'],
                                y=[1] * len(flagged_df),
                                mode='markers',
                                marker=dict(size=12, color='red', symbol='circle'),
                                name='High Risk',
                                text=flagged_df['text'].str[:50] + '...',
                                hovertemplate='%{x}<br>%{text}<extra></extra>'
                            ))
                            fig.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis_title="Date",
                                yaxis=dict(showticklabels=False),
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True, key="risk_timeline_chart")
                        else:
                            st.info("No risk data available")
                    
                    with col2:
                        st.markdown("#### Time-of-Day Risk Pattern")
                        if 'suicide_score' in df.columns:
                            flagged_hours = df[df['suicide_score'] >= 0.5]['hour']
                            
                            fig = go.Figure(data=[
                                go.Histogram(
                                    x=flagged_hours,
                                    nbinsx=24,
                                    marker_color='red'
                                )
                            ])
                            fig.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis_title="Hour of Day",
                                yaxis_title="Flagged Entries"
                            )
                            st.plotly_chart(fig, use_container_width=True, key="risk_hour_histogram")
                        else:
                            st.info("No risk data available")
                    
                    st.markdown("---")
                    
                    # === 3) Context Impact ===
                    st.markdown("### 🎯 Context Impact Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Average Sentiment by Context")
                        context_sentiment = []
                        for idx, row in df.iterrows():
                            for tag in row['tags'] if row['tags'] else []:
                                context_sentiment.append({'context': tag, 'sentiment': row['sentiment_score']})
                        
                        if context_sentiment:
                            context_df = pd.DataFrame(context_sentiment)
                            context_avg = context_df.groupby('context')['sentiment'].mean().reset_index().sort_values('sentiment', ascending=False)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=context_avg['context'],
                                    y=context_avg['sentiment'],
                                    marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in context_avg['sentiment']]
                                )
                            ])
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Context", yaxis_title="Avg Sentiment")
                            st.plotly_chart(fig, use_container_width=True, key="context_sentiment_bar_chart")
                        else:
                            st.info("No context data available")
                    
                    with col2:
                        st.markdown("#### Sentiment Over Time by Context")
                        if context_sentiment:
                            # Get top 5 contexts
                            top_contexts = context_df['context'].value_counts().head(5).index
                            
                            fig = go.Figure()
                            for context in top_contexts:
                                context_time_data = []
                                for idx, row in df.iterrows():
                                    if row['tags'] and context in row['tags']:
                                        context_time_data.append({'date': row['date'], 'sentiment': row['sentiment_score']})
                                
                                if context_time_data:
                                    ctx_df = pd.DataFrame(context_time_data)
                                    fig.add_trace(go.Scatter(
                                        x=ctx_df['date'],
                                        y=ctx_df['sentiment'],
                                        mode='lines+markers',
                                        name=context
                                    ))
                            
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date", yaxis_title="Sentiment")
                            st.plotly_chart(fig, use_container_width=True, key="context_sentiment_line_chart")
                        else:
                            st.info("No context data available")
                    
                    st.markdown("---")
                    
                    # === 4) Emotion Analytics ===
                    st.markdown("### 🎭 Emotion Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Emotion Distribution (Last 30 Days)")
                        last_30_emotions = []
                        for idx, row in last_30_days.iterrows():
                            if row['emotions']:
                                last_30_emotions.extend(row['emotions'])
                        
                        if last_30_emotions:
                            emotion_counts = pd.Series(last_30_emotions).value_counts()
                            
                            fig = go.Figure(data=[
                                go.Pie(
                                    labels=emotion_counts.index,
                                    values=emotion_counts.values,
                                    hole=0.4
                                )
                            ])
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig, use_container_width=True, key="emotion_30day_pie_chart")
                        else:
                            st.info("No emotion data available")
                    
                    with col2:
                        st.markdown("#### All-Time Emotion Distribution")
                        all_emotions_ever = []
                        for idx, row in df.iterrows():
                            if row['emotions']:
                                all_emotions_ever.extend(row['emotions'])
                        
                        if all_emotions_ever:
                            emotion_counts_all = pd.Series(all_emotions_ever).value_counts()
                            
                            fig = go.Figure(data=[
                                go.Pie(
                                    labels=emotion_counts_all.index,
                                    values=emotion_counts_all.values,
                                    hole=0.4
                                )
                            ])
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig, use_container_width=True, key="emotion_alltime_pie_chart")
                        else:
                            st.info("No emotion data available")
                    
                    st.markdown("---")
                    
                    # === 5) Language Cues ===
                    st.markdown("### 💬 Language Cues & Patterns")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Common Keywords")
                        # Extract common words (excluding stopwords)
                        from collections import Counter
                        import re
                        
                        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}
                        
                        all_words = []
                        for text in df['text']:
                            words = re.findall(r'\b[a-z]+\b', text.lower())
                            all_words.extend([w for w in words if w not in stopwords and len(w) > 3])
                        
                        if all_words:
                            word_counts = Counter(all_words).most_common(15)
                            words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=words_df['Count'],
                                    y=words_df['Word'],
                                    orientation='h',
                                    marker_color='lightblue'
                                )
                            ])
                            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Frequency", yaxis_title="")
                            st.plotly_chart(fig, use_container_width=True, key="keyword_frequency_chart")
                        else:
                            st.info("Not enough text data")
                    
                    with col2:
                        st.markdown("#### Sentiment vs Entry Length")
                        fig = go.Figure(data=[
                            go.Scatter(
                                x=df['word_count'],
                                y=df['sentiment_score'],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=df['sentiment_score'],
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    colorbar=dict(title="Sentiment")
                                ),
                                text=df['date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='Words: %{x}<br>Sentiment: %{y}<br>Date: %{text}<extra></extra>'
                            )
                        ])
                        fig.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Word Count",
                            yaxis_title="Sentiment Score"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="sentiment_length_scatter")
                    
                    st.markdown("---")
                    
                    # === 6) Routines & Patterns ===
                    st.markdown("### 🗓️ Routines & Patterns")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Calendar Heatmap - Entry Count")
                        daily_count = df.groupby(df['date'].dt.date).size().reset_index(name='count')
                        daily_count.columns = ['date', 'count']
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=daily_count['count'],
                            x=daily_count['date'],
                            y=['Entries'] * len(daily_count),
                            colorscale='Blues',
                            showscale=True
                        ))
                        fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Date")
                        st.plotly_chart(fig, use_container_width=True, key="calendar_heatmap_chart")
                    
                    with col2:
                        st.markdown("#### Sentiment Streaks")
                        
                        # Calculate streaks
                        positive_streak_current = 0
                        negative_streak_current = 0
                        neutral_streak_current = 0
                        
                        for score in df.sort_values('date', ascending=False)['sentiment_score']:
                            if score > 0:
                                positive_streak_current += 1
                                break
                            else:
                                break
                        
                        for score in df.sort_values('date', ascending=False)['sentiment_score']:
                            if score < 0:
                                negative_streak_current += 1
                                break
                            else:
                                break
                        
                        streak_data = pd.DataFrame({
                            'Streak Type': ['Longest Positive', 'Current Positive', 'Current Negative'],
                            'Days': [max_positive_streak, positive_streak_current, negative_streak_current]
                        })
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=streak_data['Streak Type'],
                                y=streak_data['Days'],
                                marker_color=['green', 'lightgreen', 'red']
                            )
                        ])
                        fig.update_layout(height=150, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Days")
                        st.plotly_chart(fig, use_container_width=True, key="sentiment_streaks_chart")
                    
                    # Hour x Day of Week Heatmap
                    st.markdown("#### Hour × Day of Week - Average Sentiment")
                    heatmap_data = df.groupby(['day_of_week', 'hour'])['sentiment_score'].mean().unstack(fill_value=0)
                    
                    # Reorder days
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        zmid=0
                    ))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Hour of Day", yaxis_title="Day of Week")
                    st.plotly_chart(fig, use_container_width=True, key="hour_dayofweek_heatmap")
                    
                    st.markdown("---")
                    
                    # === Writing Patterns ===
                    st.markdown("### ✍️ Writing Patterns")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_length = df['text_length'].mean()
                        st.metric("Average Text Length", f"{avg_length:.0f} characters")
                    with col2:
                        avg_words = df['word_count'].mean()
                        st.metric("Average Word Count", f"{avg_words:.0f} words")
                    with col3:
                        total_entries = len(df)
                        st.metric("Total Entries", total_entries)
                    
                else:
                    st.info("📝 No entries found. Add some diary entries to see detailed analytics!")
            else:
                st.info("📝 No data file found. Add some diary entries to see detailed analytics!")
                
        except Exception as e:
            st.error(f"❌ Error generating analytics: {e}")
            import traceback
            st.code(traceback.format_exc())
        
    except Exception as e:
        st.error(f"❌ Error loading statistics: {e}")

def about_page():
    """About page."""
    st.markdown("""
    <div class="feature-card">
        <h2>ℹ️ About MindLens</h2>
        <p>Your AI-powered digital diary with intelligent emotion detection and insights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <h2>📖 About Us</h2>
        <p><b>Name:</b> GOWTHAM.J</p>
        <p><b>College:</b> VIT Vellore </p>
        <p><b>Project:</b> LTIMindtree Collaboration</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="summary-box">
        <h3>🎯 Mission</h3>
        <p>{ABOUT_TEXT}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### 🌟 Project Details
    This project, <b>MindLens - AI-Powered Digital Diary</b>, is developed as part of an 
    academic–industry collaboration with <b>LTIMindtree</b>.  

    **Key Highlights:**
    - 📌 Developed by **GOWTHAM.J** at **VIT Vellore**.  
    - 🧠 AI-powered system that detects **emotions** and **sentiments** from daily diary entries.  
    - 🏷️ Adds **context & achievement tags** for better understanding of personal growth.  
    - 🔍 Uses **Sentence-BERT embeddings** stored in **FAISS/ChromaDB** for semantic search.  
    - 🤖 Supports **RAG-based querying**, enabling natural summaries grounded in past entries.  
    - 📊 Provides insightful **Streamlit dashboards** (mood trends, emotion distribution, productivity patterns).  

    **Why this matters?**
    - Helps individuals gain clarity on their mental state.  
    - Provides structured reflection on achievements and challenges.  
    - Bridges technology with personal well-being through explainable AI.  

    ---
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎭 Emotion Detection</h4>
            <p>Automatically detects emotions in your diary entries using advanced AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Semantic Search</h4>
            <p>Find entries using natural language queries and intelligent matching</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Privacy-First</h4>
            <p>All processing happens locally on your device for maximum privacy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🏷️ Context Tagging</h4>
            <p>Identifies topics and contexts in your writing automatically</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI Summaries</h4>
            <p>Get insights about patterns in your thoughts and feelings</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Analytics</h4>
            <p>Track your emotional patterns and personal growth over time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Technical Details")
    st.markdown("""
    <div class="feature-card">
        <h4>🧠 AI Models</h4>
        <p><strong>Emotion Model:</strong> Fine-tuned BERT model for emotion classification</p>
        <p><strong>Context Model:</strong> SpaCy-based text categorization</p>
        <p><strong>Embeddings:</strong> Sentence-BERT for semantic similarity</p>
        <p><strong>Vector Store:</strong> FAISS for efficient similarity search</p>
        <p><strong>LLM:</strong> Google Gemini for AI-powered summaries</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎨 Design Philosophy")
    st.markdown("""
    <div class="summary-box">
        <h3>💡 Our Vision</h3>
        <p>MindLens combines the power of AI with the intimacy of personal journaling. 
        We believe that understanding your emotions and patterns can lead to better 
        self-awareness and personal growth. Our goal is to make AI-powered insights 
        accessible while maintaining complete privacy and control over your personal data.</p>
    </div>
    """, unsafe_allow_html=True)

def view_entries_page():
    """Page for viewing all diary entries."""
    st.markdown("""
    <div class="feature-card">
        <h2>📚 View All Entries</h2>
        <p>Browse through all your diary entries stored in the database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Get all entries from the vector store
        # Since we don't have a direct method to get all entries, we'll read from the JSONL file
        entries_file = Path("data/entries.jsonl")
        
        if not entries_file.exists():
            st.info("📝 No entries found. Start by adding your first diary entry!")
            return
        
        # Read all entries
        entries = []
        with open(entries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        if not entries:
            st.info("📝 No entries found. Start by adding your first diary entry!")
            return
        
        # Sort entries by date (newest first)
        entries.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        st.markdown(f"### 📊 Total Entries: {len(entries)}")
        
        # Add search/filter options
        col1, col2 = st.columns(2)
        with col1:
            search_text = st.text_input("🔍 Search in entries", placeholder="Search for specific text...")
        with col2:
            date_filter = st.date_input("📅 Filter by date", value=None)
        
        # Filter entries
        filtered_entries = entries
        if search_text:
            filtered_entries = [e for e in filtered_entries if search_text.lower() in e.get('text', '').lower()]
        if date_filter:
            filtered_entries = [e for e in filtered_entries if e.get('date') == date_filter.isoformat()]
        
        st.markdown(f"### 📋 Showing {len(filtered_entries)} entries")
        
        # Delete options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Delete All Entries", type="secondary", use_container_width=True):
                st.session_state.show_delete_all_confirm = True
        
        with col2:
            if st.button("📝 Select Entry to Delete", type="secondary", use_container_width=True):
                st.session_state.show_delete_selection = True
        
        # Delete all confirmation
        if st.session_state.get('show_delete_all_confirm', False):
            st.markdown("""
            <div class="delete-warning">
                <h3>⚠️ WARNING: This will permanently delete ALL diary entries!</h3>
                <p>This action cannot be undone. All your diary entries will be lost forever.</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
                    try:
                        # Delete the entries file
                        entries_file.unlink()
                        # Also delete FAISS files
                        faiss_index = Path("data/faiss_index.faiss")
                        faiss_meta = Path("data/faiss_meta.jsonl")
                        if faiss_index.exists():
                            faiss_index.unlink()
                        if faiss_meta.exists():
                            faiss_meta.unlink()
                        
                        # Reload the vector store to refresh the in-memory index
                        try:
                            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'vector_store'):
                                st.session_state.app.vector_store.reload()
                        except Exception as reload_error:
                            print(f"Warning: Could not reload vector store: {reload_error}")
                        
                        st.success("✅ All entries deleted successfully!")
                        st.session_state.show_delete_all_confirm = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting entries: {e}")
            
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_delete_all_confirm = False
                st.rerun()
        
        # Delete selection mode
        if st.session_state.get('show_delete_selection', False):
            st.markdown("""
            <div class="delete-info">
                <h3>📝 Select an entry to delete</h3>
                <p>Click the "Delete This Entry" button below any entry you want to remove.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("❌ Cancel Selection", use_container_width=True):
                st.session_state.show_delete_selection = False
                st.rerun()
        
        # Display entries
        for i, entry in enumerate(filtered_entries, 1):
            with st.expander(f"📝 Entry {i} - {entry.get('date', 'Unknown Date')}", expanded=False):
                st.markdown(f"""
                <div class="entry-card">
                    <h4>📅 {entry.get('date', 'Unknown Date')}</h4>
                    <p><strong>Text:</strong> {entry.get('text', 'No text available')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show suicide risk if available
                if entry.get('suicide_score') is not None:
                    suicide_score = entry.get('suicide_score', 0.0)
                    suicide_prediction = entry.get('suicide_prediction', 'Unknown')
                    if suicide_score >= 0.5:
                        risk_color = "#ff6b6b"
                        risk_level = "High Risk"
                    else:
                        risk_color = "#4caf50"
                        risk_level = "Low Risk"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 6px solid {risk_color}; margin: 1rem 0;">
                        <h3>🛡️ Mental Health Risk</h3>
                        <h2 style="color: {risk_color};">{risk_level}</h2>
                        <p style="font-size: 0.8em; margin: 0;">Score: {suicide_score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show image if available
                if entry.get('image_path'):
                    st.markdown("### 🖼️ Image")
                    try:
                        st.image(str(entry.get('image_path')))
                        if entry.get('image_desc'):
                            st.caption(f"Description: {entry.get('image_desc')}")
                    except Exception:
                        st.info("Image associated but could not be previewed.")
                
                # Show video if available
                if entry.get('video_path'):
                    st.markdown("### 🎥 Video")
                    try:
                        video_path = Path(entry.get('video_path'))
                        if video_path.exists():
                            st.video(str(video_path))
                        else:
                            st.warning(f"⚠️ Video file not found: {entry.get('video_path')}")
                    except Exception as e:
                        st.info(f"Video associated but could not be previewed: {e}")
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>😊 Sentiment</h3>
                        <h2>{entry.get('sentiment', 'Unknown').title()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    emotions = entry.get('emotions', [])
                    emotion_text = ', '.join(emotions) if emotions else 'None'
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎭 Emotions</h3>
                        <h2>{emotion_text}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    tags = entry.get('tags', [])
                    tag_text = ', '.join(tags) if tags else 'None'
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏷️ Tags</h3>
                        <h2>{tag_text}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show emotions and tags as badges
                if emotions:
                    st.markdown("### 🎭 Emotions")
                    emotion_html = ""
                    for emotion in emotions:
                        emotion_html += f'<span class="emotion-badge">{emotion}</span>'
                    st.markdown(emotion_html, unsafe_allow_html=True)
                
                if tags:
                    st.markdown("### 🏷️ Tags")
                    tag_html = ""
                    for tag in tags:
                        tag_html += f'<span class="tag-badge">{tag}</span>'
                    st.markdown(tag_html, unsafe_allow_html=True)
                
                # Delete individual entry option
                if st.session_state.get('show_delete_selection', False):
                    st.markdown("---")
                    entry_id = entry.get('doc_id', '')
                    if st.button(f"🗑️ Delete This Entry", key=f"delete_{entry_id}", type="secondary", use_container_width=True):
                        try:
                            # Remove entry from all files
                            delete_entry_from_files(entry_id, entries_file)
                            st.success("✅ Entry deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting entry: {e}")
    
    except Exception as e:
        st.error(f"❌ Error loading entries: {e}")

def delete_entry_from_files(entry_id, entries_file):
    """Delete a specific entry from all storage files and rebuild FAISS index."""
    try:
        # Read all entries
        entries = []
        if entries_file.exists():
            with open(entries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get('doc_id') != entry_id:
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        
        # Rewrite entries file without the deleted entry
        with open(entries_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Update FAISS metadata file and rebuild index
        faiss_meta_file = Path("data/faiss_meta.jsonl")
        faiss_index_file = Path("data/faiss_index.faiss")
        
        if faiss_meta_file.exists():
            faiss_entries = []
            with open(faiss_meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            if entry.get('doc_id') != entry_id:
                                faiss_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            
            # Rewrite FAISS metadata
            with open(faiss_meta_file, 'w', encoding='utf-8') as f:
                for entry in faiss_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Rebuild FAISS index from scratch
            if faiss_entries:
                import numpy as np
                import faiss
                
                # Extract embeddings
                embeddings = []
                for entry in faiss_entries:
                    if 'embedding' in entry and entry['embedding']:
                        embeddings.append(entry['embedding'])
                
                if embeddings:
                    vecs = np.array(embeddings).astype('float32')
                    d = vecs.shape[1]
                    index = faiss.IndexFlatIP(d)
                    index.add(vecs)
                    faiss.write_index(index, str(faiss_index_file))
            else:
                # No entries left, remove index file
                if faiss_index_file.exists():
                    faiss_index_file.unlink()
        
        # Reload the vector store to refresh the in-memory index
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'vector_store'):
                st.session_state.app.vector_store.reload()
        except Exception as reload_error:
            print(f"Warning: Could not reload vector store: {reload_error}")
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to delete entry: {e}")

def download_pdf_page():
    """Page for downloading diary entries as PDF."""
    st.markdown("""
    <div class="feature-card">
        <h2>📄 Download PDF</h2>
        <p>Export all your diary entries as a beautiful PDF document.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Get all entries
        entries_file = Path("data/entries.jsonl")
        
        if not entries_file.exists():
            st.info("📝 No entries found to download. Start by adding some diary entries!")
            return
        
        # Read all entries
        entries = []
        with open(entries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        if not entries:
            st.info("📝 No entries found to download. Start by adding some diary entries!")
            return
        
        # Sort entries by date
        entries.sort(key=lambda x: x.get('date', ''))
        
        st.markdown(f"""
        <div class="download-section">
            <h3>📊 Ready to Download</h3>
            <p>Found {len(entries)} diary entries ready for PDF export</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PDF generation options
        col1, col2 = st.columns(2)
        with col1:
            include_metadata = st.checkbox("📊 Include emotions and tags", value=True)
        with col2:
            page_size = st.selectbox("📄 Page size", ["A4", "Letter"])
        
        if st.button("📥 Generate & Download PDF", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating PDF..."):
                try:
                    # Create PDF
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4 if page_size == "A4" else letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Title
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=24,
                        spaceAfter=30,
                        alignment=1,  # Center alignment
                        textColor=colors.HexColor('#667eea')
                    )
                    story.append(Paragraph("MindLens - Digital Diary", title_style))
                    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
                    story.append(Spacer(1, 20))
                    
                    # Add entries
                    for i, entry in enumerate(entries, 1):
                        # Entry header
                        date_str = entry.get('date', 'Unknown Date')
                        story.append(Paragraph(f"Entry {i} - {date_str}", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        # Entry text
                        text = entry.get('text', 'No text available')
                        story.append(Paragraph(text, styles['Normal']))
                        story.append(Spacer(1, 12))
                        
                        # Metadata if requested
                        if include_metadata:
                            sentiment = entry.get('sentiment', 'Unknown')
                            emotions = ', '.join(entry.get('emotions', [])) or 'None'
                            tags = ', '.join(entry.get('tags', [])) or 'None'
                            
                            metadata = f"""
                            <b>Sentiment:</b> {sentiment.title()}<br/>
                            <b>Emotions:</b> {emotions}<br/>
                            <b>Tags:</b> {tags}
                            """
                            story.append(Paragraph(metadata, styles['Normal']))
                        
                        story.append(Spacer(1, 20))
                        
                        # Add page break every 3 entries (except for the last one)
                        if i % 3 == 0 and i < len(entries):
                            story.append(PageBreak())
                    
                    # Build PDF
                    doc.build(story)
                    buffer.seek(0)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download PDF",
                        data=buffer.getvalue(),
                        file_name=f"mindlens_diary_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF generated successfully! Click the download button above.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")
    
    except Exception as e:
        st.error(f"❌ Error loading entries: {e}")

def mental_support_page():
    """Page for finding nearby mental health support services."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    .mental-health-header {
        background: linear-gradient(135deg, #E8E4F3 0%, #F0F8FF 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(139, 69, 19, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .mental-health-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 300;
        color: #6B5B95;
        margin: 0 0 1rem 0;
        letter-spacing: -0.02em;
    }
    .mental-health-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: #8B7AA8;
        margin: 0;
        line-height: 1.6;
    }
    .mental-health-encouragement {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #A8A2B8;
        margin: 1rem 0 0 0;
        font-style: italic;
    }
    </style>
    
    <div class="mental-health-header">
        <h1 class="mental-health-title">🌸 Mental Health Support</h1>
        <p class="mental-health-subtitle">Find nearby mental health professionals, counseling centers, and support resources in your area.</p>
        <p class="mental-health-encouragement">You're not alone — help is available and you deserve support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Location input with calm, minimal styling
    st.markdown("""
    <style>
    .location-section {
        background: linear-gradient(135deg, #F8F6FF 0%, #F0F8FF 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .location-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 400;
        color: #6B5B95;
        margin: 0 0 0.5rem 0;
    }
    .location-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #8B7AA8;
        margin: 0;
    }
    </style>
    
    <div class="location-section">
        <h3 class="location-title">📍 Enter Your Location</h3>
        <p class="location-subtitle">We'll search for mental health services near this location</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add more spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_location = st.text_input(
            "Location (e.g., Vellore, Tamil Nadu)",
            placeholder="Enter your city, state, or area",
            help="We'll search for mental health services near this location",
            key="location_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with input
        search_button = st.button("🔍 Find Support", use_container_width=True, type="primary")
    
    if search_button and user_location.strip():
        with st.spinner("🔍 Searching for mental health services near you..."):
            try:
                # Find nearby hospitals
                hospitals = mental_health_service.find_nearby_hospitals(user_location.strip())
                
                if hospitals:
                    st.markdown("""
                    <style>
                    .hospitals-header {
                        background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%);
                        padding: 2rem;
                        border-radius: 20px;
                        margin: 2rem 0;
                        text-align: center;
                        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.06);
                        border: 1px solid rgba(255, 255, 255, 0.8);
                    }
                    .hospitals-title {
                        font-family: 'Inter', sans-serif;
                        font-size: 1.6rem;
                        font-weight: 400;
                        color: #6B5B95;
                        margin: 0 0 0.5rem 0;
                    }
                    .hospitals-count {
                        font-family: 'Inter', sans-serif;
                        font-size: 0.95rem;
                        font-weight: 300;
                        color: #8B7AA8;
                        margin: 0;
                    }
                    .hospital-card {
                        background: linear-gradient(135deg, #FFFFFF 0%, #F8F6FF 100%);
                        margin: 1.5rem 0;
                        padding: 2rem;
                        border-radius: 20px;
                        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.08);
                        border: 1px solid rgba(255, 255, 255, 0.8);
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                    }
                    .hospital-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 8px 32px rgba(107, 91, 149, 0.12);
                    }
                    .hospital-name {
                        font-family: 'Inter', sans-serif;
                        font-size: 1.3rem;
                        font-weight: 500;
                        color: #6B5B95;
                        margin: 0 0 1.5rem 0;
                        line-height: 1.4;
                    }
                    .hospital-detail {
                        font-family: 'Inter', sans-serif;
                        font-size: 0.95rem;
                        font-weight: 400;
                        color: #8B7AA8;
                        margin: 0.8rem 0;
                        line-height: 1.5;
                    }
                    .hospital-detail strong {
                        color: #6B5B95;
                        font-weight: 500;
                    }
                    .hospital-link {
                        color: #8B7AA8;
                        text-decoration: none;
                        border-bottom: 1px solid rgba(139, 122, 168, 0.3);
                        transition: color 0.2s ease;
                    }
                    .hospital-link:hover {
                        color: #6B5B95;
                    }
                    </style>
                    
                    <div class="hospitals-header">
                        <h3 class="hospitals-title">🏥 Nearby Mental Health Services</h3>
                        <p class="hospitals-count">Found {len(hospitals)} mental health services near you</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, hospital in enumerate(hospitals, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="hospital-card">
                                <h4 class="hospital-name">
                                    {i}. {hospital['name']}
                                </h4>
                                <div style="background: rgba(248, 246, 255, 0.5); padding: 1.5rem; border-radius: 16px; margin: 0;">
                                    <p class="hospital-detail"><strong>📍 Address:</strong> {hospital['address']}</p>
                                    <p class="hospital-detail"><strong>📞 Contact:</strong> {hospital['contact_number']}</p>
                                    {f'<p class="hospital-detail"><strong>🌐 Website:</strong> <a href="{hospital["website"]}" target="_blank" class="hospital-link">{hospital["website"]}</a></p>' if hospital['website'] else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.success(f"✅ Found {len(hospitals)} mental health services near {user_location}")
                else:
                    st.warning("⚠️ No specific hospitals found for your location. Please try a broader area or check the crisis resources below.")
                
            except Exception as e:
                st.error(f"❌ Error searching for hospitals: {e}")
                st.info("Please try again or check the crisis resources below.")
    
    # Crisis resources section with calm, minimal styling
    st.markdown("""
    <style>
    .crisis-header {
        background: linear-gradient(135deg, #FFE8E8 0%, #F0F8FF 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin: 3rem 0 2rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .crisis-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 400;
        color: #8B4A6B;
        margin: 0 0 0.5rem 0;
    }
    .crisis-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #A88B9B;
        margin: 0;
    }
    .crisis-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF8F8 100%);
        margin: 1.5rem 0;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(139, 74, 107, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .crisis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(139, 74, 107, 0.12);
    }
    .crisis-name {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        color: #8B4A6B;
        margin: 0 0 1rem 0;
    }
    .crisis-number {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #6B5B95;
        margin: 0 0 0.5rem 0;
        letter-spacing: 0.5px;
    }
    .crisis-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        color: #A88B9B;
        margin: 0;
        line-height: 1.4;
    }
    </style>
    
    <div class="crisis-header">
        <h3 class="crisis-title">🆘 Immediate Crisis Resources</h3>
        <p class="crisis-subtitle">If you're in immediate crisis, please contact these resources</p>
    </div>
    """, unsafe_allow_html=True)
    
    crisis_resources = mental_health_service.get_crisis_resources()
    
    col1, col2 = st.columns(2)
    
    for i, resource in enumerate(crisis_resources):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="crisis-card">
                <h4 class="crisis-name">{resource['name']}</h4>
                <p class="crisis-number">{resource['contact_number']}</p>
                <p class="crisis-description">{resource['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional support information with calm, minimal styling
    st.markdown("""
    <style>
    .support-section {
        background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 3rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .support-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 400;
        color: #6B5B95;
        margin: 0 0 2rem 0;
    }
    .support-content {
        background: rgba(255, 255, 255, 0.6);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: left;
        max-width: 600px;
        margin: 0 auto;
    }
    .support-remember {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        color: #6B5B95;
        margin: 0 0 1.5rem 0;
    }
    .support-list {
        font-family: 'Inter', sans-serif;
        color: #8B7AA8;
        font-size: 1rem;
        line-height: 1.8;
        margin: 0;
        padding: 0;
    }
    .support-list li {
        margin: 1.2rem 0;
        list-style: none;
        position: relative;
        padding-left: 1.5rem;
    }
    .support-list li:before {
        content: "✨";
        position: absolute;
        left: 0;
        top: 0;
    }
    .support-list strong {
        color: #6B5B95;
        font-weight: 500;
    }
    </style>
    
    <div class="support-section">
        <h3 class="support-title">💡 Additional Support</h3>
        <div class="support-content">
            <h4 class="support-remember">🌟 Remember:</h4>
            <ul class="support-list">
                <li><strong>You are not alone</strong> — Many people care about you and want to help</li>
                <li><strong>It's okay to ask for help</strong> — Seeking support is a sign of strength</li>
                <li><strong>This feeling is temporary</strong> — Even when it doesn't feel like it, things can get better</li>
                <li><strong>Professional help works</strong> — Mental health professionals are trained to help</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add Emergency Contact Section
    st.markdown("""
    <style>
    .add-contact-section {
        background: linear-gradient(135deg, #E8F0FF 0%, #F0F8FF 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin: 3rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(107, 91, 149, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    .add-contact-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 400;
        color: #4A6B8B;
        margin: 0 0 1.5rem 0;
    }
    .add-contact-form {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 16px;
        max-width: 500px;
        margin: 0 auto;
    }
    </style>
    
    <div class="add-contact-section">
        <h3 class="add-contact-title">📞 Add Emergency Contact</h3>
        <div class="add-contact-form">
            <p style="color: #6B7A95; margin-bottom: 1.5rem;">Add trusted contacts who can be notified in crisis situations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Form for adding emergency contacts
    with st.form("add_contact_form"):
        contact_name = st.text_input("Contact Name", placeholder="e.g., Parent, Friend, Therapist")
        contact_phone = st.text_input("Phone Number", placeholder="e.g., +919876543210 or 9876543210")
        
        # Submit button for the form
        submitted = st.form_submit_button("➕ Add Contact")
        
        if submitted:
            if contact_name.strip() and contact_phone.strip():
                # Add contact using mental_health_service
                if mental_health_service.add_emergency_contact(contact_name.strip(), contact_phone.strip()):
                    st.success(f"✅ Emergency contact '{contact_name}' added successfully!")
                    # Show current contacts
                    try:
                        contacts = mental_health_service.get_emergency_contacts()
                        if contacts:
                            st.markdown("### 📋 Current Emergency Contacts")
                            for i, contact in enumerate(contacts, 1):
                                st.markdown(f"{i}. **{contact['name']}** - {contact['phone']}")
                    except Exception as e:
                        pass  # Silently ignore if we can't display contacts
                else:
                    st.error("❌ Failed to add emergency contact. Please try again.")
            else:
                st.warning("⚠️ Please enter both name and phone number.")

def faq_page():
    """Page for FAQ.""" 
    # Import and run the FAQ page
    import importlib.util
    spec = importlib.util.spec_from_file_location("faq", "MINDLENS/pages/5_FAQ.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

if __name__ == "__main__":
    main()


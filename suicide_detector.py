
#!/usr/bin/env python3
"""
Suicide Detection Model Integration for MINDLENS
Loads a trained Keras model and tokenizer for suicide risk assessment.
"""

import os
import re
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import json


# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable TensorFlow for now due to compatibility issues
TF_AVAILABLE = False
print("⚠️ TensorFlow disabled due to compatibility issues, using fallback detector")

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class SuicideDetector:
    """Suicide risk detection using trained Keras model."""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Initialize the suicide detector.
        
        Args:
            model_path: Path to the trained Keras model (.h5 file)
            tokenizer_path: Path to the pickled tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.stop_words = set(stopwords.words('english'))
        self.max_length = 100  # Adjust based on your model's expected input length
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load the trained Keras model."""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping model loading")
            self.model = None
            return
            
        try:
            self.model = load_model(self.model_path)
            print(f"✅ Suicide detection model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def _load_tokenizer(self):
        """Load the pickled tokenizer."""
        try:
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {self.tokenizer_path}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            self.tokenizer = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for suicide detection.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove short words (length < 3)
        words = text.split()
        words = [word for word in words if len(word) >= 3]
        
        # Remove stopwords
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict suicide risk from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model or not self.tokenizer:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probability': 0.0,
                'emotion': 'Unknown',
                'tags': ['error']
            }
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'prediction': 'Non-Suicidal',
                    'confidence': 0.0,
                    'probability': 0.0,
                    'emotion': 'Neutral',
                    'tags': ['insufficient_text']
                }
            
            # Tokenize and pad
            sequences = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=self.max_length, padding='post'
            )
            
            # Make prediction
            prediction = self.model.predict(padded_sequences, verbose=0)[0][0]
            
            # Determine label and confidence
            is_suicidal = prediction >= 0.5
            label = "Suicidal" if is_suicidal else "Non-Suicidal"
            confidence = prediction if is_suicidal else (1.0 - prediction)
            
            # Infer emotion and tags
            emotion = self._infer_emotion(text, prediction)
            tags = self._extract_tags(text, prediction)
            
            return {
                'prediction': label,
                'confidence': float(confidence * 100),
                'probability': float(prediction),
                'emotion': emotion,
                'tags': tags
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'probability': 0.0,
                'emotion': 'Unknown',
                'tags': ['error']
            }
    
    def _infer_emotion(self, text: str, prediction: float) -> str:
        """Infer basic emotion from text and prediction."""
        text_lower = text.lower()
        
        # High risk emotions
        if prediction >= 0.7:
            if any(word in text_lower for word in ['hopeless', 'worthless', 'empty', 'numb']):
                return 'Despair'
            elif any(word in text_lower for word in ['angry', 'rage', 'furious', 'hate']):
                return 'Anger'
            else:
                return 'Sadness'
        
        # Medium risk emotions
        elif prediction >= 0.4:
            if any(word in text_lower for word in ['worried', 'anxious', 'scared', 'fear']):
                return 'Anxiety'
            elif any(word in text_lower for word in ['tired', 'exhausted', 'drained']):
                return 'Fatigue'
            else:
                return 'Concern'
        
        # Low risk emotions
        else:
            if any(word in text_lower for word in ['happy', 'excited', 'joy', 'great', 'wonderful']):
                return 'Happiness'
            elif any(word in text_lower for word in ['calm', 'peaceful', 'relaxed', 'content']):
                return 'Calm'
            else:
                return 'Neutral'
    
    def _extract_tags(self, text: str, prediction: float) -> List[str]:
        """Extract relevant tags based on text content and prediction."""
        text_lower = text.lower()
        tags = []
        
        # Risk level tags
        if prediction >= 0.7:
            tags.append('high_risk')
        elif prediction >= 0.4:
            tags.append('moderate_risk')
        else:
            tags.append('low_risk')
        
        # Content-based tags
        if any(word in text_lower for word in ['depression', 'depressed', 'sad', 'down']):
            tags.append('depression')
        
        if any(word in text_lower for word in ['anxiety', 'anxious', 'worried', 'nervous']):
            tags.append('anxiety')
        
        if any(word in text_lower for word in ['hopeless', 'hopelessness', 'no point', 'pointless']):
            tags.append('hopelessness')
        
        if any(word in text_lower for word in ['worthless', 'useless', 'failure', 'loser']):
            tags.append('low_self_worth')
        
        if any(word in text_lower for word in ['suicide', 'kill myself', 'end it', 'not worth living']):
            tags.append('suicidal_ideation')
        
        if any(word in text_lower for word in ['help', 'support', 'therapy', 'counseling']):
            tags.append('seeking_help')
        
        if any(word in text_lower for word in ['positive', 'good', 'great', 'excited', 'happy']):
            tags.append('positive')
        
        if any(word in text_lower for word in ['work', 'job', 'career', 'professional']):
            tags.append('work_related')
        
        if any(word in text_lower for word in ['family', 'friend', 'relationship', 'love']):
            tags.append('relationships')
        
        return tags
    
    def get_motivational_quote(self, prediction: float) -> str:
        """Get a motivational quote based on prediction score."""
        quotes = {
            'high_risk': [
                "You are stronger than you know, and you don't have to face this alone.",
                "This feeling is temporary, even if it doesn't feel that way right now.",
                "You matter, and your life has value beyond what you can see today.",
                "Reaching out for help is a sign of strength, not weakness."
            ],
            'moderate_risk': [
                "It's okay to not be okay. Take it one day at a time.",
                "Small steps forward are still progress, even when they feel small.",
                "You've overcome challenges before, and you can overcome this one too.",
                "Your feelings are valid, and it's okay to ask for support."
            ],
            'low_risk': [
                "You're doing great! Keep taking care of yourself.",
                "Every positive thought is a step in the right direction.",
                "You have the strength within you to handle whatever comes your way.",
                "Remember to celebrate the small victories along the way."
            ]
        }
        
        if prediction >= 0.7:
            category = 'high_risk'
        elif prediction >= 0.4:
            category = 'moderate_risk'
        else:
            category = 'low_risk'
        
        import random
        return random.choice(quotes[category])
    
    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded successfully."""
        return self.model is not None and self.tokenizer is not None
    
    def send_whatsapp_alert(self, contacts_file: str, message: str) -> bool:
        """
        Send WhatsApp alert to emergency contacts.
        
        Args:
            contacts_file: Path to JSON file containing emergency contacts
            message: Message to send
            
        Returns:
            True if alerts were sent successfully, False otherwise
        """
        try:
            import pywhatkit as kit
            
            # Check if contacts file exists
            if not os.path.exists(contacts_file):
                print(f"⚠️ Emergency contacts file not found: {contacts_file}")
                return False
            
            # Load emergency contacts
            with open(contacts_file, 'r') as f:
                contacts = json.load(f)
            
            if not contacts:
                print("⚠️ No emergency contacts found")
                return False
            
            # Send message to each contact
            success_count = 0
            for contact in contacts:
                phone = contact.get('phone')
                name = contact.get('name', 'Emergency Contact')
                
                if not phone:
                    print(f"⚠️ Phone number missing for contact: {name}")
                    continue
                
                # Normalize phone number
                if not phone.startswith('+'):
                    phone = '+91' + phone
                
                try:
                    # Send instantly (no scheduling)
                    # Using getattr to avoid linter issues
                    send_func = getattr(kit, 'sendwhatmsg_instantly')
                    send_func(
                        phone_no=phone,
                        message=message,
                        wait_time=10,   # seconds to wait for WhatsApp Web to load
                        tab_close=True, # close the tab automatically
                        close_time=3    # seconds before closing tab
                    )
                    print(f"✅ WhatsApp alert sent to {name} ({phone})")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Failed to send WhatsApp alert to {name} ({phone}): {e}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error sending WhatsApp alerts: {e}")
            return False

    def send_whatsapp_alert_to_phone(self, phone_number: str, message: str) -> bool:
        """
        Send WhatsApp alert to a specific phone number (used for user-provided numbers not in database).
        
        Args:
            phone_number: Phone number to send alert to
            message: Message to send
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            import pywhatkit as kit
            
            # Validate phone number
            if not phone_number:
                print("⚠️ Phone number is required")
                return False
            
            # Normalize phone number
            if not phone_number.startswith('+'):
                phone_number = '+91' + phone_number
            
            print(f"📨 Sending WhatsApp alert to {phone_number}...")
            
            try:
                # Send instantly (no scheduling)
                # Using getattr to avoid linter issues
                send_func = getattr(kit, 'sendwhatmsg_instantly')
                send_func(
                    phone_no=phone_number,
                    message=message,
                    wait_time=10,   # seconds to wait for WhatsApp Web to load
                    tab_close=True, # close the tab automatically
                    close_time=3    # seconds before closing tab
                )
                print(f"✅ WhatsApp alert sent to {phone_number}")
                return True
                
            except Exception as e:
                print(f"❌ Failed to send WhatsApp alert to {phone_number}: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Error sending WhatsApp alert: {e}")
            return False

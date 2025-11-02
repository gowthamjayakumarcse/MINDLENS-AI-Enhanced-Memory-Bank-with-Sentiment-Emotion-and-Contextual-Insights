#!/usr/bin/env python3
"""
Fallback Suicide Detection Model Integration for MINDLENS
This version works without loading the actual Keras model, using rule-based analysis.
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

class SuicideDetectorFallback:
    """Fallback suicide risk detection using rule-based analysis."""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Initialize the fallback suicide detector.
        
        Args:
            model_path: Path to the trained Keras model (.h5 file) - not used in fallback
            tokenizer_path: Path to the pickled tokenizer - not used in fallback
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        
        # High-risk keywords and phrases
        self.high_risk_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living', 'want to die',
            'end my life', 'take my life', 'hurt myself', 'self harm', 'cut myself',
            'overdose', 'jump off', 'hang myself', 'shoot myself', 'poison myself',
            'no point', 'hopeless', 'worthless', 'useless', 'failure', 'loser',
            'hate myself', 'deserve to die', 'better off dead', 'world without me',
            'nobody cares', 'everyone hates me', 'burden', 'waste of space',
            'dont see point', 'see any point', 'point continuing', 'falling apart',
            'better off without me', 'everyone would be better', 'hate myself'
        ]
        
        # Medium-risk keywords
        self.medium_risk_keywords = [
            'depressed', 'sad', 'down', 'empty', 'numb', 'lonely', 'isolated',
            'anxious', 'worried', 'scared', 'afraid', 'panic', 'overwhelmed',
            'tired', 'exhausted', 'drained', 'burned out', 'stressed',
            'angry', 'rage', 'furious', 'hate', 'resent', 'bitter',
            'confused', 'lost', 'directionless', 'purposeless', 'aimless',
            'struggling', 'depression', 'dark thoughts', 'struggling depression',
            'struggling with', 'having dark', 'dark thoughts',
            'feeling down', 'feeling sad', 'feeling empty', 'feeling numb',
            'cant cope', 'cant handle', 'too much', 'overwhelming',
            'dont know what', 'dont know how', 'lost hope', 'losing hope'
        ]
        
        # Low-risk/positive keywords
        self.positive_keywords = [
            'happy', 'excited', 'joy', 'great', 'wonderful', 'amazing', 'fantastic',
            'good', 'better', 'improving', 'progress', 'success', 'achievement',
            'proud', 'confident', 'hopeful', 'optimistic', 'positive', 'upbeat',
            'grateful', 'thankful', 'blessed', 'lucky', 'fortunate', 'content',
            'will pass', 'know it will', 'temporary', 'feeling better', 'getting better',
            'looking forward', 'excited about', 'can handle', 'will get through'
        ]
        
        # Stop words for text processing
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        print("✅ Fallback suicide detector initialized (rule-based analysis)")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
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
        Predict suicide risk from text using rule-based analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
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
            
            # Count keyword matches
            text_lower = text.lower()
            high_risk_count = sum(1 for keyword in self.high_risk_keywords if keyword in text_lower)
            medium_risk_count = sum(1 for keyword in self.medium_risk_keywords if keyword in text_lower)
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            
            # Calculate risk score with better weighting
            total_risk = high_risk_count * 4 + medium_risk_count * 2  # Increased medium risk weight
            total_positive = positive_count * 2
            
            # Normalize score (0-1 range)
            if total_risk + total_positive == 0:
                probability = 0.0
            else:
                probability = min(1.0, total_risk / (total_risk + total_positive + 1))
            
            # Adjust for high-risk keywords
            if high_risk_count > 0:
                probability = min(1.0, probability + 0.4)  # Increased boost for high-risk
            
            # Adjust for medium-risk keywords (ensure they contribute meaningfully)
            if medium_risk_count > 0 and high_risk_count == 0:
                # For medium-risk only, use more conservative scoring
                if medium_risk_count == 1:
                    probability = min(0.4, max(0.2, probability))  # Single medium-risk keyword
                else:
                    probability = min(0.6, max(0.3, probability))  # Multiple medium-risk keywords
            
            # Determine label and confidence
            is_suicidal = probability >= 0.5
            label = "Suicidal" if is_suicidal else "Non-Suicidal"
            confidence = probability if is_suicidal else (1.0 - probability)
            
            # Infer emotion and tags
            emotion = self._infer_emotion(text, probability)
            tags = self._extract_tags(text, probability)
            
            return {
                'prediction': label,
                'confidence': float(confidence * 100),
                'probability': float(probability),
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
        """Check if detector is ready (always true for fallback)."""
        return True
    
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
            import time
            
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
            
            # Remove duplicate contacts based on phone number
            unique_contacts = []
            seen_phones = set()
            for contact in contacts:
                phone = contact.get('phone')
                if phone:
                    # Normalize phone number for comparison
                    normalized_phone = phone if phone.startswith('+') else '+91' + phone
                    if normalized_phone not in seen_phones:
                        unique_contacts.append(contact)
                        seen_phones.add(normalized_phone)
            
            if len(unique_contacts) != len(contacts):
                print(f"⚠️ Removed {len(contacts) - len(unique_contacts)} duplicate contact(s)")
            
            contacts = unique_contacts
            print(f"📨 Sending WhatsApp alerts to {len(contacts)} unique contact(s)...")
            
            # Send message to each contact with delay between sends
            success_count = 0
            for i, contact in enumerate(contacts):
                phone = contact.get('phone')
                name = contact.get('name', 'Emergency Contact')
                
                if not phone:
                    print(f"⚠️ Phone number missing for contact: {name}")
                    continue
                
                # Normalize phone number
                if not phone.startswith('+'):
                    phone = '+91' + phone
                
                try:
                    print(f"Sending to {name} ({phone})...")
                    
                    # Send instantly (no scheduling)
                    # Using getattr to avoid linter issues
                    send_func = getattr(kit, 'sendwhatmsg_instantly')
                    send_func(
                        phone_no=phone,
                        message=message,
                        wait_time=15,    # seconds to wait for WhatsApp Web to load
                        tab_close=True,  # close the tab automatically
                        close_time=5     # seconds before closing tab
                    )
                    print(f"✅ WhatsApp alert sent to {name} ({phone})")
                    success_count += 1
                    
                    # Add delay between contacts to prevent multiple tabs opening simultaneously
                    # Only delay if not the last contact
                    if i < len(contacts) - 1:
                        print("Waiting for tab to close before sending to next contact...")
                        time.sleep(8)  # Wait for previous tab to close (wait_time + close_time + buffer)
                    
                except Exception as e:
                    print(f"❌ Failed to send WhatsApp alert to {name} ({phone}): {e}")
            
            if success_count > 0:
                print(f"✅ Successfully sent alerts to {success_count} contact(s)")
            
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
            import time
            
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
                    wait_time=15,    # seconds to wait for WhatsApp Web to load
                    tab_close=True,  # close the tab automatically
                    close_time=5     # seconds before closing tab
                )
                print(f"✅ WhatsApp alert sent to {phone_number}")
                return True
                
            except Exception as e:
                print(f"❌ Failed to send WhatsApp alert to {phone_number}: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Error sending WhatsApp alert: {e}")
            return False

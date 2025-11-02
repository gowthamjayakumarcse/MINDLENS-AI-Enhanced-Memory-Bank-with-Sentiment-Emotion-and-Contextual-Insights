#!/usr/bin/env python3
"""
MindLens - AI-Powered Digital Diary
Main application file that integrates all components.
"""

import os
# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import (
    GOEMOTIONS_MODEL_DIR, SPACY_MODEL_DIR, SBERT_MODEL_DIR,
    VECTOR_STORE, LLM_BACKEND, HF_API_TOKEN, HF_MODEL,
    SUICIDE_MODEL_PATH, SUICIDE_TOKENIZER_PATH,
    EMERGENCY_CONTACTS_JSON, AUTO_ALERT_ENABLED
)
from emotion_model import EmotionModel
from tagger import ContextTagger
from embedder import SBERTEmbedder
from storage import VectorStore, DiaryRecord, new_doc_id
from sentiment_rules import votes_to_sentiment
# Use fallback detector due to TensorFlow compatibility issues
print("Using fallback rule-based suicide detector...")
from suicide_detector_fallback import SuicideDetectorFallback as SuicideDetector
from rag import summarize_hits
from utils import parse_date_str

class MindLensApp:
    """Main application class for MindLens digital diary."""
    
    def __init__(self):
        """Initialize all models and components."""
        print("Initializing MindLens...")
        
        # Initialize models
        print("Loading emotion detection model...")
        self.emotion_model = EmotionModel(GOEMOTIONS_MODEL_DIR)
        
        print("Loading context tagging model...")
        self.tagger = ContextTagger(SPACY_MODEL_DIR)
        
        print("Loading embedding model...")
        self.embedder = SBERTEmbedder(SBERT_MODEL_DIR)
        
        print("Initializing vector store...")
        self.vector_store = VectorStore()
        
        print("Loading suicide detection model...")
        self.suicide_detector = SuicideDetector(SUICIDE_MODEL_PATH, SUICIDE_TOKENIZER_PATH)
        
        # Store config for access in UI
        import config
        self.config = config
        
        print("MindLens initialized successfully!")
    
    def process_entry(self, text: str, date: Optional[str] = None,
                      image_path: Optional[str] = None,
                      image_desc: Optional[str] = None,
                      video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a diary entry and store it in the vector database.
        
        Args:
            text: The diary entry text
            date: Optional date string (defaults to today)
            image_path: Optional path to image file
            image_desc: Optional image description
            video_path: Optional path to video file
            
        Returns:
            Dictionary with processing results
        """
        if date is None:
            date = datetime.now().date().isoformat()
        else:
            date = parse_date_str(date)
        
        print(f"Processing entry for {date}...")
        
        # Generate embeddings (include image description if provided)
        text_for_embedding = text
        if image_desc:
            text_for_embedding = f"{text}\n[Image Description]: {image_desc}"
        embedding = self.embedder.encode([text_for_embedding])[0].tolist()
        
        # Detect emotions
        emotions = self.emotion_model.predict(text)
        
        # Extract context tags with lower threshold for better detection
        tags = self.tagger.predict(text, top_k=5, threshold=0.1)
        
        # Determine sentiment from emotions
        sentiment = votes_to_sentiment(emotions)
        
        # Detect suicide risk
        suicide_result = self.suicide_detector.predict(text)
        
        # Create diary record
        doc_id = new_doc_id()
        record = DiaryRecord(
            doc_id=doc_id,
            date=date,
            text=text,
            embedding=embedding,
            sentiment=sentiment,
            emotions=emotions,
            tags=tags,
            image_path=image_path,
            image_desc=image_desc,
            video_path=video_path
        )
        
        # Store in vector database
        self.vector_store.upsert([record])
        
        result = {
            "doc_id": doc_id,
            "date": date,
            "sentiment": sentiment,
            "emotions": emotions,
            "tags": tags,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "image_path": image_path,
            "image_desc": image_desc,
            "video_path": video_path,
            "suicide_score": suicide_result.get('probability', 0.0),
            "suicide_prediction": suicide_result.get('prediction', 'Unknown'),
            "suicide_confidence": suicide_result.get('confidence', 0.0),
            "suicide_emotion": suicide_result.get('emotion', 'Unknown'),
            "suicide_tags": suicide_result.get('tags', [])
        }
        
        print(f"Entry processed and stored. ID: {doc_id}")
        return result
    
    def search_entries(self, query: str, top_k: int = 5, 
                      filter_emotions: Optional[List[str]] = None,
                      filter_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search diary entries using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_emotions: Optional list of emotions to filter by
            filter_tags: Optional list of tags to filter by
            
        Returns:
            List of matching diary entries
        """
        print(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Build filter conditions
        where_conditions = {}
        if filter_emotions:
            where_conditions["emotions"] = {"$contains": filter_emotions}
        if filter_tags:
            where_conditions["tags"] = {"$contains": filter_tags}
        
        # Search vector store
        hits = self.vector_store.query(
            query_vec=query_embedding,
            top_k=top_k,
            where=where_conditions if where_conditions else None
        )
        
        print(f"Found {len(hits)} matching entries")
        return hits
    
    def get_ai_summary(self, query: str, top_k: int = 5) -> str:
        """
        Get AI-powered summary of search results.
        
        Args:
            query: Search query
            top_k: Number of entries to include in summary
            
        Returns:
            AI-generated summary
        """
        hits = self.search_entries(query, top_k)
        if not hits:
            return "No matching entries found."
        
        if LLM_BACKEND == "none":
            return self._format_simple_summary(query, hits)
        
        return summarize_hits(query, hits)
    
    def _format_simple_summary(self, query: str, hits: List[Dict[str, Any]]) -> str:
        """Format a simple summary without LLM."""
        lines = [f"Query: {query}", "", "Top matches:"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. [{h.get('date')}] {h.get('text')}")
            emo = ", ".join(h.get("emotions", []))
            tags = ", ".join(h.get("tags", []))
            lines.append(f"   emotions: {emo} | sentiment: {h.get('sentiment')} | tags: {tags}")
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored entries."""
        # This would require implementing a method to count entries
        # For now, return basic info
        return {
            "vector_store": VECTOR_STORE,
            "llm_backend": LLM_BACKEND,
            "models_loaded": {
                "emotion_model": True,
                "context_tagger": True,
                "embedder": True
            }
        }

def main():
    """Main function for command-line usage."""
    app = MindLensApp()
    
    print("\n" + "="*50)
    print("MindLens - AI-Powered Digital Diary")
    print("="*50)
    print("Commands:")
    print("1. add <text> - Add a diary entry")
    print("2. search <query> - Search entries")
    print("3. summary <query> - Get AI summary")
    print("4. stats - Show statistics")
    print("5. quit - Exit")
    print("="*50)
    
    while True:
        try:
            command = input("\nEnter command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif command.startswith('add '):
                text = command[4:].strip()
                if text:
                    result = app.process_entry(text)
                    print(f"Added entry: {result}")
                else:
                    print("Please provide text for the entry.")
            
            elif command.startswith('search '):
                query = command[7:].strip()
                if query:
                    results = app.search_entries(query)
                    for i, result in enumerate(results, 1):
                        print(f"{i}. [{result.get('date')}] {result.get('text')}")
                        print(f"   Emotions: {', '.join(result.get('emotions', []))}")
                        print(f"   Tags: {', '.join(result.get('tags', []))}")
                        print(f"   Sentiment: {result.get('sentiment')}")
                        print()
                else:
                    print("Please provide a search query.")
            
            elif command.startswith('summary '):
                query = command[8:].strip()
                if query:
                    summary = app.get_ai_summary(query)
                    print(f"AI Summary:\n{summary}")
                else:
                    print("Please provide a query for summary.")
            
            elif command == 'stats':
                stats = app.get_stats()
                print("Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()


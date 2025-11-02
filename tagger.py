
from typing import List
import spacy
from pathlib import Path
import re

class ContextTagger:
    def __init__(self, model_dir: str):
        model_dir = str(Path(model_dir))
        try:
            self.nlp = spacy.load(model_dir)
            # Check if the model has labels
            textcat = self.nlp.get_pipe("textcat_multilabel")
            if textcat.labels:
                self.use_model = True
                print(f"Using spacy model with {len(textcat.labels)} labels")
            else:
                self.use_model = False
                print("Spacy model has no labels, using keyword-based tagging")
        except Exception as e:
            print(f"Error loading spacy model: {e}, using keyword-based tagging")
            self.use_model = False
        
        # Define the expected categories based on the training data
        self.expected_categories = [
            "work", "study_learning", "meeting_social", "achievement", "family", 
            "relationships", "leisure_entertainment", "routine_chores", "travel_commute", 
            "finance_money", "creativity", "productivity", "health", "self_care", 
            "stress_mental_state", "goals_planning", "reflection_journaling", 
            "deadline", "appointment", "important_dates"
        ]
        
        # Keyword-based tagging fallback
        self.keyword_mapping = {
            "work": ["work", "job", "office", "meeting", "presentation", "boss", "colleague", "project", "deadline", "business"],
            "study_learning": ["study", "learn", "school", "university", "college", "class", "lecture", "homework", "exam", "test", "book", "course"],
            "family": ["family", "mother", "father", "mom", "dad", "sister", "brother", "parent", "relative", "aunt", "uncle"],
            "health": ["health", "doctor", "hospital", "medicine", "sick", "ill", "pain", "exercise", "gym", "fitness", "medical"],
            "relationships": ["friend", "boyfriend", "girlfriend", "partner", "relationship", "love", "dating", "marriage", "wedding"],
            "leisure_entertainment": ["movie", "film", "music", "game", "party", "fun", "entertainment", "hobby", "sport", "travel", "vacation"],
            "routine_chores": ["cook", "cooking", "clean", "cleaning", "grocery", "shopping", "laundry", "dishes", "housework"],
            "travel_commute": ["travel", "trip", "flight", "train", "bus", "car", "drive", "commute", "journey", "vacation"],
            "finance_money": ["money", "pay", "salary", "budget", "expensive", "cheap", "buy", "purchase", "cost", "price", "bank"],
            "creativity": ["art", "creative", "design", "write", "writing", "draw", "paint", "music", "poetry", "craft"],
            "productivity": ["productive", "efficient", "organize", "plan", "schedule", "task", "goal", "achieve", "complete", "finish"],
            "self_care": ["relax", "rest", "sleep", "meditation", "yoga", "spa", "massage", "peaceful", "calm", "mindful"],
            "stress_mental_state": ["stress", "stressed", "anxiety", "worried", "nervous", "tired", "exhausted", "overwhelmed", "pressure"],
            "goals_planning": ["goal", "plan", "future", "dream", "ambition", "career", "success", "achieve", "target", "objective"],
            "reflection_journaling": ["think", "thought", "reflect", "journal", "diary", "contemplate", "consider", "ponder", "meditate"],
            "deadline": ["deadline", "due", "urgent", "rush", "hurry", "time", "schedule", "appointment", "meeting"],
            "appointment": ["appointment", "meeting", "schedule", "doctor", "dentist", "interview", "date", "time"],
            "important_dates": ["birthday", "anniversary", "holiday", "celebration", "event", "special", "important", "milestone"]
        }

    def predict(self, text: str, top_k: int = 5, threshold: float = 0.3) -> List[str]:
        if self.use_model:
            # Try to use the spacy model first
            doc = self.nlp(text)
            cats = getattr(doc, "cats", {})
            
            # Filter to only include expected categories and apply threshold
            filtered_cats = {cat: score for cat, score in cats.items() 
                            if cat in self.expected_categories and score >= threshold}
            
            # Sort by score and return top_k
            sorted_cats = sorted(filtered_cats.items(), key=lambda x: -x[1])
            tags = [cat for cat, score in sorted_cats[:top_k]]
            
            if tags:  # If model returned results, use them
                return tags
        
        # Fallback to keyword-based tagging
        text_lower = text.lower()
        tag_scores = {}
        
        for tag, keywords in self.keyword_mapping.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                tag_scores[tag] = score / len(keywords)  # Normalize by number of keywords
        
        # Sort by score and return top_k
        sorted_tags = sorted(tag_scores.items(), key=lambda x: -x[1])
        tags = [tag for tag, score in sorted_tags[:top_k] if score >= threshold]
        
        return tags

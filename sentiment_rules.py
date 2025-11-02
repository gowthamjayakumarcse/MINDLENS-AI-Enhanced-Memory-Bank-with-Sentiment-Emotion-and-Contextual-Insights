
# Rule-based mapping from GoEmotions labels to sentiment buckets.
# Customize to your liking.
EMOTION_TO_SENTIMENT = {
    # Positive
    "joy": "positive",
    "admiration": "positive",
    "amusement": "positive",
    "approval": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "love": "positive",
    "optimism": "positive",
    "pride": "positive",
    "relief": "positive",
    "desire": "positive",
    "caring": "positive",
    "curiosity": "positive",
    "surprise": "positive",  # can be neutral/pos; keep positive

    # Negative
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "embarrassment": "negative",
    "fear": "negative",
    "grief": "negative",
    "nervousness": "negative",
    "remorse": "negative",
    "sadness": "negative",
    "disgust": "negative",
    "confusion": "negative",
    "realization": "neutral",  # treat as neutral-ish
    "surprise_negative": "negative",  # if your model has custom variants
    "anxiety": "negative",  # if custom label exists

    # Neutralish
    "neutral": "neutral",
    "boredom": "neutral",
}
def votes_to_sentiment(emotions):
    if not emotions:
        return "neutral"
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for e in emotions:
        s = EMOTION_TO_SENTIMENT.get(e.lower(), "neutral")
        counts[s] += 1
    # majority vote, tie -> neutral
    if counts["positive"] > counts["negative"] and counts["positive"] > counts["neutral"]:
        return "positive"
    if counts["negative"] > counts["positive"] and counts["negative"] > counts["neutral"]:
        return "negative"
    return "neutral"

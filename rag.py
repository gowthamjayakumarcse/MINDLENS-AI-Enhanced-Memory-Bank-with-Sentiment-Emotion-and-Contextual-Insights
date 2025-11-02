from typing import List, Dict, Any

# Import configuration
try:
    from config import LLM_BACKEND, HF_API_TOKEN, HF_MODEL
except ImportError:
    LLM_BACKEND = "none"
    HF_API_TOKEN = ""
    HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def summarize_hits(query: str, hits: List[Dict[str, Any]]) -> str:
    """Summarize search results using configured LLM backend."""
    if not hits:
        return "No matching entries found."
    
    # Route to appropriate backend
    if LLM_BACKEND == "huggingface":
        return _summarize_with_hf_llm(query, hits)
    elif LLM_BACKEND == "none":
        return _format_simple_summary(query, hits)
    else:
        return _format_simple_summary(query, hits)


def _summarize_with_hf_llm(query: str, hits: List[Dict[str, Any]]) -> str:
    """Summarize using Hugging Face Inference API."""
    try:
        from huggingface_hub import InferenceClient
        
        # Create the API client
        client = InferenceClient(model=HF_MODEL, token=HF_API_TOKEN)
        
        # Build the summarization prompt
        prompt = _build_prompt(query, hits)
        
        print("🤖 Generating summary using Hugging Face API...")
        
        # Use chat completion for conversational models like Llama
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Call the remote model using chat completion
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            top_p=0.9,
        )
        
        # Extract the generated text from the response
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                result = content.strip()
                if result:
                    print("✅ Summary generated successfully!")
                    return result
        
        print("⚠️ LLM returned empty output.")
        return _format_simple_summary(query, hits)
            
    except Exception as e:
        print(f"❌ Error using Hugging Face API: {e}")
        print("Falling back to simple formatted summary.")
        return _format_simple_summary(query, hits)


def _format_simple_summary(query: str, hits: List[Dict[str, Any]]) -> str:
    """Fallback: simple extractive summary if API fails."""
    lines = [
        f"📝 Summary for: {query}",
        "",
        f"Found {len(hits)} relevant entries:",
        "",
    ]

    for i, h in enumerate(hits, 1):
        date = h.get("date", "Unknown date")
        text = h.get("text", "")
        emotions = ", ".join(h.get("emotions", []))
        tags = ", ".join(h.get("tags", []))
        sentiment = h.get("sentiment", "neutral")

        lines.append(f"{i}. [{date}] {text[:150]}{'...' if len(text) > 150 else ''}")
        lines.append(f"   🎭 Emotions: {emotions}")
        lines.append(f"   😊 Sentiment: {sentiment}")
        lines.append(f"   🏷️ Tags: {tags}")
        lines.append("")

    return "\n".join(lines)


def _build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    """Build the prompt for summarization with diary context."""
    context = ""
    for h in hits:
        date = h.get("date", "Unknown")
        text = h.get("text", "")
        emotions = ", ".join(h.get("emotions", []))
        tags = ", ".join(h.get("tags", []))
        sentiment = h.get("sentiment", "neutral")

        context += (
            f"- [{date}] {text} "
            f"(emotions: {emotions}; tags: {tags}; sentiment: {sentiment})\n"
        )

    return f"""You are MindLens, an empathetic and insightful diary assistant.

User Query: {query}

Below are the most relevant diary excerpts retrieved for the query. 
Analyze them carefully and produce a thoughtful, coherent summary that addresses the user's request.

Your task:
1. Identify recurring **patterns, themes, and emotional tones** across the excerpts (e.g., anxiety, regret, hope, relief).
2. Reference **specific dates** when they appear (e.g., “On 2025-10-12…”). 
   - If no exact date is given, use relative phrasing (e.g., “in an earlier entry,” “later that week”).
3. Provide **empathetic insight** — validate the writer’s emotions, highlight growth, and note emotional shifts or coping strategies.
4. Keep the response **concise but meaningful** (2–3 paragraphs of natural, reflective prose).
5. **Do not fabricate** details or events not supported by the excerpts.

Tone & Style:
- Warm, respectful, and emotionally intelligent.
- Avoid clinical language or judgment.
- Write in fluent, human-like narrative paragraphs (no bullet points unless essential).
- Use short quotes from the diary only when they enrich the interpretation.

Diary Excerpts:
{context}

Your Summary:

"""

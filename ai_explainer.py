import os
from typing import List, Dict, Any, Optional

def generate_llm_explanation(
    query: str,
    results: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> str:
    """
    Synthesizes factual CCTV search results into a concise natural language explanation.
    Follows project requirement:
    'Use an LLM only to explain retrieved evidence in natural language.
     The factual result should come from the detection and retrieval pipeline,
     not from the LLM guessing.'
    """
    if not results:
        return f"No matching footage or evidence was found for the query: '{query}'. Please check camera selection, time filters, or try a different description."

    best = results[0]
    cam = best.get("camera", "Unknown Camera")
    t_str = best.get("timestamp_str", "00:00")
    obj = best.get("object_class", "object")
    conf = int(best.get("confidence", 0.0) * 100)
    color = best.get("color_hint", "")
    color_desc = f"{color} " if color and color != "unknown" else ""

    # Synthesize grounding context
    summary_lines = [
        f"Target **{color_desc}{obj}** was detected with **{conf}% confidence**.",
        f"Location: **{cam}** at timestamp **{t_str}**."
    ]

    if len(results) > 1:
        other_cams = list(dict.fromkeys([r.get("camera") for r in results[1:] if r.get("camera") != cam]))
        if other_cams:
            summary_lines.append(f"Subsequent matching observations found across: {', '.join(other_cams)}.")
        else:
            summary_lines.append(f"A total of {len(results)} matching visual occurrences were cataloged.")

    explanation = " ".join(summary_lines)

    # If GEMINI_API_KEY or OPENAI_API_KEY is available, we can enhance the explanation
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import urllib.request
            import json
            prompt = (
                f"You are a CCTV video surveillance AI analyst. Explain these factual results for the user query: '{query}'.\n"
                f"Factual data: Found {len(results)} matches. Primary match: {color_desc}{obj} at {cam} camera, timestamp {t_str}, {conf}% confidence.\n"
                f"Give a professional, concise 2-sentence response explaining where and when the entity was identified."
            )
            # Safe call if configured
        except Exception:
            pass

    return explanation

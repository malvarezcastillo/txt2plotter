"""Stage 1: Prompt engineering via OpenRouter LLM."""

import os

from openai import OpenAI

from .utils import save_debug

SYSTEM_PROMPT = """You rewrite user prompts for Flux.2 image generation,
optimized for pen plotter line art output.

Flux.2 uses natural language - write flowing descriptions, not keyword lists.
Word order matters: put the most important elements first.

Structure: Subject + Style + Details + Mood

ALWAYS frame as line art by including phrases like:
- "minimalistic line drawing" or "clean line art"
- "black ink on white paper" or "monochrome ink illustration"
- "clean precise lines" or "pen and ink style"
- "technical illustration" or "architectural line drawing"

DO NOT use:
- "single continuous line", "one unbroken line", or "single stroke" - Flux.2 cannot
  reliably produce true continuous line drawings; use general line art styles instead
- Negative phrasing ("no shading", "without color") - Flux has no negative prompts
- Keyword spam - use natural sentences instead
- "white background" phrase - causes blurry outputs

Example transformation:
Input: "a geometric skull"
Output: "Minimalistic line drawing of a geometric skull composed of
triangular facets and sharp angular planes, black ink on white paper,
technical illustration style with clean precise lines,
symmetrical front view, high contrast monochrome"

Output ONLY the rewritten prompt."""


def enhance_prompt(user_prompt: str) -> str:
    """Enhance a user prompt for Flux.2 line art generation.

    Args:
        user_prompt: The user's original prompt.

    Returns:
        Enhanced prompt optimized for Flux.2 line art output.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://github.com/txt2plotter",
            "X-Title": "txt2svg",
        },
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    )

    enhanced = response.choices[0].message.content.strip()
    save_debug("01_prompt_enhanced.txt", f"Original: {user_prompt}\n\nEnhanced: {enhanced}")

    return enhanced

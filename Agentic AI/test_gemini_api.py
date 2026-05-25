"""One-shot Gemini connectivity check. Run from project root: python test_gemini_api.py"""
import google.generativeai as genai

from config import GEMINI_MODEL, require_api_key


def main() -> None:
    key = require_api_key()
    genai.configure(api_key=key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content("Reply with exactly: API OK")
    except Exception as e:
        print("Model:", GEMINI_MODEL)
        print("Error:", e)
        print(
            "Tip: set GEMINI_MODEL in .env to a model your project has quota for "
            "(e.g. gemini-1.5-flash or gemini-2.0-flash)."
        )
        raise
    text = (response.text or "").strip()
    print("Model:", GEMINI_MODEL)
    print("Response:", text)
    if not text:
        print("Warning: empty text; check model name / safety / finish_reason:", response.candidates)


if __name__ == "__main__":
    main()

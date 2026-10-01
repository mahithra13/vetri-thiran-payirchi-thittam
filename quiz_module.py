import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def clean_json_block(text: str) -> str:
    return re.sub(r"```(?:json)?\n([\s\S]*?)```", r"\1", text).strip()

def generate_quiz(text: str) -> list:
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        prompt = f"""
You are a quiz generator.
From the following passage, create 3 multiple-choice questions. Each question should include:
- A "question"
- A list of 4 "options"
- A correct "answer" that must exactly match one of the options.
Format your output as **valid JSON**, like this:
[
  {{
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]
Passage:
{text}
"""
        response = model.generate_content(prompt)
        quiz_text = response.text.strip()
        cleaned_text = clean_json_block(quiz_text)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Quiz compilation error: {e}")
        return []
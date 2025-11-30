import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No API Key found in .env")
    exit()

genai.configure(api_key=api_key)

print(f"Checking models for key: {api_key[:5]}...")

try:
    print("Available Models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
    print("\nAPI Key is working!")
except Exception as e:
    print(f"\nError: {e}")
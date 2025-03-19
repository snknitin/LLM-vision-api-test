# LLM-vision-api-test


Setup Instructions

Set up your environment:
```bashCopy
pip install streamlit pillow google-generativeai python-dotenv requests
```

Create a .env file with your API keys:
```Copy
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

Choose the implementation based on your preferred model:

* The Gemini implementation is more cost-effective for high volume processing
* The GPT-4o implementation provides more detailed analysis but at higher cost
* You can integrate both and let users switch between them
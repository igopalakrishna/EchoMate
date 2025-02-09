# EchoMate

AI Mood & Stress Management Assistant

This repository contains a Gradio-based AI assistant that helps users improve their mood and reduce stress using OpenAI GPT-4o, Whisper, and text-to-speech (TTS) functionalities.

Features

Voice Input: Speak to the AI and receive transcriptions using Whisper.

Mood Analysis: AI assesses user emotions and stress levels.

Relaxation Suggestions: Provides personalized relaxation exercises.

Text-to-Speech (TTS): AI responses are spoken aloud.

Gradio UI: Simple web-based interface for interaction.

Installation & Setup

1. Clone the Repository

2. Install Dependencies

Install all required dependencies from requirements.txt:

pip install -q -r requirements.txt

3. Store API Key in .env

Create a .env file in the project directory and add your OpenAI API key:

echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

Alternatively, manually create .env and add:

OPENAI_API_KEY=your_openai_api_key_here

4. Run the Application

Start the Gradio app:

python app.py

This will launch the AI assistant in your browser.

Usage

Speak into the microphone to interact with the AI.

The AI will transcribe, analyze, and respond accordingly.

If stressed, the AI suggests relaxation techniques.

Responses are spoken aloud using TTS.

Deployment

To deploy this application, consider using:

Hugging Face Spaces (for hosting Gradio apps)

# Imports
import gradio as gr

import os
import json
import whisper

from dotenv import load_dotenv
from openai import OpenAI

from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

import ffmpeg 

import torch

# Initialization
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are an empathetic AI companion designed to help users improve their mood and reduce stress."
system_message += " Begin by asking the user about their current emotional state in a warm and non-intrusive manner."
system_message += " Ask one thoughtful and relevant question at a time based on the user's responses."
system_message += " Tailor your responses and suggestions based on the user's mood, emotional intensity and offer practical yet gentle recommendations, such as mindfulness techniques, or positive affirmations."
system_message += " Maintain a compassionate and understanding tone throughout the conversation."
system_message += " Conclude the session naturally when the user seems satisfied or ready to end the conversation."
system_message += " Summarize the key points discussed and provide words of encouragement before ending the session."
system_message += " Politely indicate the end of the session and remind the user that they can return anytime for more support."

mood_check_function = {
    "name": "mood_check",
    "description": "Ask the user to reflect on their mood and emotions to better understand their current emotional state.",
    "parameters": {
        "type": "object",
        "properties": {
            "current_mood": {
                "type": "string",
                "description": "he user's current mood, such as 'calm', 'anxious', 'happy', 'frustrated', 'stressed', 'lonely' or 'excited'."
            },
            "emotion_strength": {
                "type": "string",
                "enum": ["mild", "moderate", "strong"],
                "description": "The intensity of the emotion being experienced."
            },
        },
        "required": ["current_mood", "emotion_strength"],
        "additionalProperties": False
    }
}

stress_assessment_function = {
    "name": "stress_level_assessment",
    "description": "If the user is stressed then assess the user’s stress level based on their current feelings and external stressors.",
    "parameters": {
        "type": "object",
        "properties": {
            "stress_cause": {
                "type": "string",
                "description": "An explanation of the user’s current stressor, such as work, family, health."
            },
            "stress_intensity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "The intensity of the user’s perceived stress."
            }
        },
        "required": ["stress_cause", "stress_intensity"],
        "additionalProperties": False
    }
}

stress_assessment_function = {
    "name": "stress_level_assessment",
    "description": "If the user is stressed then assess the user’s stress level based on their current feelings and external stressors.",
    "parameters": {
        "type": "object",
        "properties": {
            "stress_cause": {
                "type": "string",
                "description": "An explanation of the user’s current stressor, such as work, family, health."
            },
            "stress_intensity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "The intensity of the user’s perceived stress."
            }
        },
        "required": ["stress_cause", "stress_intensity"],
        "additionalProperties": False
    }
}

exercise_suggestion_function = {
    "name": "relaxation_exercise_suggestion",
    "description": "Provide a personalized relaxation exercise based on the user's current mood and stress levels.",
    "parameters": {
        "type": "object",
        "properties": {
            "mood_state": {
                "type": "string",
                "description": "The user’s current emotional state, such as 'stressed', 'calm', or 'anxious'."
            },
            "stress_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "The level of stress the user is experiencing."
            },
            "exercise_type": {
                "type": "string",
                "enum": ["breathing", "meditation", "physical", "visualization"],
                "description": "The type of relaxation exercise to suggest."
            }
        },
        "required": ["mood_state", "stress_level", "exercise_type"],
        "additionalProperties": False
    }
}

positive_affirmation_function = {
    "name": "positive_affirmation",
    "description": "Offer a positive affirmation to help the user shift their mindset and reduce negative thoughts.",
    "parameters": {
        "type": "object",
        "properties": {
            "affirmation_type": {
                "type": "string",
                "enum": ["self-worth", "calmness", "strength", "hope"],
                "description": "The type of affirmation to provide based on the user’s needs."
            },
            "personalization_details": {
                "type": "string",
                "description": "Additional details about the user to personalize the affirmation (e.g., 'You are doing your best' or 'You are capable of handling challenges')."
            }
        },
        "required": ["affirmation_type", "personalization_details"],
        "additionalProperties": False
    }
}

reflection_function = {
    "name": "reflection_prompt",
    "description": "Encourage the user to reflect on their day or experience to better understand their emotions and thoughts.",
    "parameters": {
        "type": "object",
        "properties": {
            "reflection_type": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "The type of reflection prompt to offer (e.g., focus on positive experiences, identify sources of negativity)."
            },
            "focus_area": {
                "type": "string",
                "description": "The area of focus for reflection, such as 'work', 'relationships', or 'personal growth'."
            }
        },
        "required": ["reflection_type", "focus_area"],
        "additionalProperties": False
    }
}


concluding_function = {
    "name": "conclude_session",
    "description": "Check if user wants to continue the mood and stress management session or conclude it with personlised insights.",
    "parameters": {
        "type": "object",
        "properties": {
            "continue_session": {
                "type": "boolean",
                "description": "Indicates if the user wants to continue the session. If false, provide conclusion."
            },
            "session_summary": {
                "type": "string",
                "description": "A brief summary of the user's mood and emotions during the session, including key takeaways."
            },
            "suggestions_for_improvement": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Suggestions to help the user improve their mood or reduce stress, personalized based on the user's emotional state."
                },
                "description": "Personalized recommendations for improving mental well-being, such as exercises, activities, or self-care tips."
            },
            "encouragement": {
                "type": "string",
                "description": "A message of encouragement or affirmation to help the user feel supported and motivated moving forward."
            }
        },
        "required": ["continue_session"],
        "additionalProperties": False
    }
}


concluding_function = {
    "name": "conclude_session",
    "description": "Check if user wants to continue the mood and stress management session or conclude it with personlised insights.",
    "parameters": {
        "type": "object",
        "properties": {
            "continue_session": {
                "type": "boolean",
                "description": "Indicates if the user wants to continue the session. If false, provide conclusion."
            },
            "session_summary": {
                "type": "string",
                "description": "A brief summary of the user's mood and emotions during the session, including key takeaways."
            },
            "suggestions_for_improvement": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Suggestions to help the user improve their mood or reduce stress, personalized based on the user's emotional state."
                },
                "description": "Personalized recommendations for improving mental well-being, such as exercises, activities, or self-care tips."
            },
            "encouragement": {
                "type": "string",
                "description": "A message of encouragement or affirmation to help the user feel supported and motivated moving forward."
            }
        },
        "required": ["continue_session"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": mood_check_function},
    {"type": "function", "function": stress_assessment_function},
    {"type": "function", "function": exercise_suggestion_function},
    {"type": "function", "function": positive_affirmation_function},
    {"type": "function", "function": reflection_function},
    {"type": "function", "function": concluding_function},
]


def handle_tool_call(message):
    """Handle tool calls based on the AI response."""
    tool_call = message.tool_calls[0]  # Assuming one tool call at a time
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    response_content = {}

    if function_name == "mood_check_function":
        mood = arguments.get("current_mood")
        intensity = arguments.get("emotion_strength")
        response_content = {"response": f"You are feeling {mood} with {intensity} intensity."}
        
    elif function_name == "stress_assessment_function":
        stress_cause = arguments.get("stress_cause")
        stress_intensity = arguments.get("stress_intensity")
        response_content = {"assessment": f"Your stress is caused by {stress_cause} with a {stress_intensity} intensity."}

    elif function_name == "exercise_suggestion_function":
        mood_state = arguments.get("mood_state")
        stress_level = arguments.get("stress_level")
        exercise_type = arguments.get("exercise_type")
        response_content = {"suggestion": f"Since you're feeling {mood_state} with {stress_level} stress, try a {exercise_type} exercise."}

    elif function_name == "positive_affirmation_function":
        affirmation_type = arguments.get("affirmation_type")
        personalization_details = arguments.get("personalization_details")
        response_content = {"affirmation": f"Here’s a {affirmation_type} affirmation for you: {personalization_details}"}

    elif function_name == "reflection_function":
        reflection_type = arguments.get("reflection_type")
        focus_area = arguments.get("focus_area")
        response_content = {"reflection": f"Let’s reflect on {focus_area} with a {reflection_type} approach."}
    
    elif function_name == "concluding_function":
        continue_session = arguments.get("continue_session")

        if(continue_session):
            response_content = {"message": "Glad to hear you want to continue! Let me know what you are feeling and what you want to focus on next."}
        else:
            session_summary = arguments.get("session_summary")
            suggestions = arguments.get("suggestions_for_improvement", [])
            encouragement = arguments.get("encouragement")
            response_content = {
                "conclusion": f"Session Summary: {session_summary}. Suggestions: {', '.join(suggestions)}. Encouragement: {encouragement}"}

    response = {
        "role": "tool",
        "content": json.dumps(response_content),
        "tool_call_id": message.tool_calls[0].id
    }
    
    return response


from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

def talker(message):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="onyx",    
      input=message
    )
    
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)

import whisper 

model  = whisper.load_model("tiny.en")

def transcribe(audio_file):
    speech_to_text = model.transcribe(audio_file)["text"]

    return speech_to_text

import whisper 

model  = whisper.load_model("tiny.en")

def transcribe(audio_file):
    speech_to_text = model.transcribe(audio_file)["text"]

    return speech_to_text

def handle_audio(audio_file, history):
    """Handle user voice input, transcribe it, and provide an audio response."""
    if audio_file is not None:
        try:
            # Transcribe the audio
            text = transcribe(audio_file)
            
            # Update history with user input
            history.append({"role": "user", "content": text})
            
            # Generate AI response
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_message}] + history,
                tools = tools
            )

            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                response = handle_tool_call(message)
                messages.append(message)
                messages.append(response)
                response = openai.chat.completions.create(model=MODEL, messages=messages)
            
            # Access the AI response message content
            reply = response.choices[0].message.content
            
            # Update history with AI response
            history.append({"role": "assistant", "content": reply})
            
            # Respond using text-to-speech
            talker(reply)
            
            return history  # Return updated chatbot display
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)  # Clean up temporary file

        return history

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages", label="AI Assistant")  # Chatbot to display conversation

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Speak to AI")  # Audio input for user voice

    with gr.Row():
        clear = gr.Button("Clear")

    # Audio input handling
    audio_input.stop_recording(
        handle_audio, 
        inputs=[audio_input, chatbot], 
        outputs=chatbot
    ).then(
        lambda history: history, 
        inputs=[chatbot], 
        outputs=chatbot
    )

    # Clear button to reset history
    clear.click(lambda: ([{"role": "system", "content": system_message}], [{"role": "system", "content": system_message}]), 
                inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)

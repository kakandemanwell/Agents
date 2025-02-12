import google.generativeai as genai
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Ensure API key is set
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")

# Model configuration
model = "gemini-2.0-pro-exp-02-05"
generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

# Initialize the Api
genai.configure(api_key=api_key)

# function to call the model
def PromptCall(prompt: str, history: list = None):
    """
    Sends a prompt to Gemini-2.0-pro-exp-02-05 and returns the response.

    Args:
        prompt (str): The input query for the model.
        history (list, optional): A list of previous interactions for contextual conversations.

    Returns:
        str: The model's response.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-pro-exp-02-05",
            generation_config=generation_config,
        )
        chat_session = model.start_chat(history=history or [])

        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    output = PromptCall(prompt)
    print("\nGemini Response:\n", output)
import os
import openai
from dotenv import load_dotenv
# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Template with placeholder
SUMMARY_PROMPT_TEMPLATE = """
You are a helpful assistant. Summarize the following
text into 3 concise bullet points:
---
{text}
---
Bullet Points:
"""
EMAIL_FIX_TEMPLATE = """
Revise the following email to sound more professional,
but keep it short and polite:
---
{text}
---Rewritten Email:
"""
def run_summary_pipeline(input_text):
    # Inject user input into the prompt
    prompt = SUMMARY_PROMPT_TEMPLATE.format(text=input_text)
    # Call OpenAI's completion endpoint (GPT-3.5 or GPT-4)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": prompt}],
        temperature=0.7, # creativity:  low = accurate, less creative; high = more creative
        max_tokens=300
    ) 
    return response['choices'][0]['message']['content']

def generate_with_prompt(template, input_text,
    role="assistant", temperature=0.5):
    prompt = template.format(text=input_text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": f"You are a helpful {role}."},
        {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=500
    ) 
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("Paste your text below to summarize:\n")
    user_input = input()
    output = run_summary_pipeline(user_input)
    print("\n Summary:\n", output)

    print("\n Revised Email:\n", generate_with_prompt(EMAIL_FIX_TEMPLATE, user_input))

    '''
    “Artificial intelligence (AI) is transforming industries by automating tasks, analyzing data at scale, and improving decision-making.
Generative AI, specifically, is unlocking new capabilities in content creation, such as text, images, and even music. As enterprises
adopt these tools, it’s important to balance innovation with ethical considerations.”
    '''
    
    '''
    Summary:
- AI is revolutionizing industries through automation and data analysis.
- Generative AI enables creation of diverse content like text and images.
- Ethical concerns must be addressed as adoption grows.
    '''
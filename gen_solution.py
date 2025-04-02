import openai
import requests

def gpt4o(prompt):
    # Set your API key
    client = openai.Client(api_key = "your key here")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[{"role": "system", "content": "Please make a complete python code solution for the following problem. Do not include tests or excessive comments."}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating solution: {e}")
        return ""


def deepseek(prompt):
    # Set your API key
    client = openai.Client(api_key = "your key here", base_url = "https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "Please make a complete python code solution for the following problem. Do not include tests or excessive comments."},
            {"role": "user", "content": prompt},
        ],
        stream=False
)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating solution: {e}")
        return ""
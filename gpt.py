import requests

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"


headers = {"Authorization": "Bearer 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"}  # Replace YOUR_TOKEN with your token
def get_answer(question):
    response = requests.post(API_URL, headers=headers, json={"inputs": question, "parameters": {"max_length": 3000}})

    result = response.json()
    print("DEBUG Response:", result)  # For debugging

    # Safely access the response
    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]['generated_text']
    elif "error" in result:
        return f"Error from API: {result['error']}"
    else:
        return "Unexpected response format. Could be a model loading issue."

print(get_answer("Can you explain what CNN is in neural networks and what it's used for in VERY VERY detail?"))

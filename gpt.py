import openai

openai.api_key = "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"  # Replace with your OpenAI API key

question = "What is Python used for?"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "user", "content": question}
    ]
)

answer = response['choices'][0]['message']['content']
print("Answer:", answer)

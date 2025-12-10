from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyCaUV2qDsfJJiD1N9ABhVa421zYWav_qJA",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ]
)

print(response.choices[0].message)

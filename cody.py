from openai import OpenAI

client = OpenAI(

    base_url="https://cody.su/api/v1",

    api_key="cody-NPyRFCZ7Pm5pNjIuwMb655VvAGopee0w1FIxqmA0b4GElHEOfU6uU8PhNnUbPn4GQMUYC63Wwwf4-qjjN9KWAQ",
    )




completion = client.chat.completions.create(

    model="grok-3",

    messages=[

        {"role": "user", "content": "Привет, напиши историю про 2х кошек, с именем Соня и Алиса"}

    ]
    )




print(completion.choices[0].message.content)



from openai import OpenAI
print(OpenAI(base_url="https://cody.su/api/v1", api_key="cody-NPyRFCZ7Pm5pNjIuwMb655VvAGopee0w1FIxqmA0b4GElHEOfU6uU8PhNnUbPn4GQMUYC63Wwwf4-qjjN9KWA").models.list()
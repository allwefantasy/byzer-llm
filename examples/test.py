from google import genai
import asyncio
import os
async def run():
    api_key = os.environ["MODEL_GOOGLE_TOKEN"]
    client = genai.Client(api_key=api_key)

    chat = client.aio.chats.create(
        model='gemini-2.5-pro-exp-03-25',  # or gemini-2.0-flash-thinking-exp
    )
    response = await chat.send_message('generativeai sdk gemini 如何获得 thoughts 部分?')
    print(response)
    print(response.candidates)
    print(response.text)    

asyncio.run(run())
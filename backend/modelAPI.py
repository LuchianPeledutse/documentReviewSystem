import os
import json
import uuid
import requests

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# prompts

# TOKENS
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

app = FastAPI()

class TokenRequest(BaseModel):
    text: str

class TokenResponse(BaseModel):
    token: str

def generate_tokens(prompt: str, access_token: str) -> str:
    rq_uid = str(uuid.uuid4())
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    payload = json.dumps({
    "model": "GigaChat",
    "messages": [
        {
        "role": "user",
        "content": prompt,
        }
    ],
    "temperature": 1,
    "top_p": 0.1,
    "stream": False,
    "n": 1,
    "max_tokens": 2048,
    "repetition_penalty": 1,
    "update_interval": 0
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    "RqUID": rq_uid,
    'Authorization': f'Bearer {access_token}'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    return json.loads(response.text)["choices"][0]["message"]["content"]

@app.post("/generate-tokens", response_model = TokenResponse)
def create_token(request: TokenRequest):
    answer = generate_tokens(request.text)
    return TokenResponse(token=answer)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
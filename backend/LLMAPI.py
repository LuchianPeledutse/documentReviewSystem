import os
import sys
import json
import uuid
import requests
from typing import List
sys.path.append("..\\")

from systemComponents import LlmModelModule
from prompts import suggestions_based_on_technologies

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel



AUTH_TOKEN = os.getenv("AUTH_TOKEN")

def get_access_token(auth_token):
  url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

  payload = 'scope=GIGACHAT_API_PERS'
  headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'RqUID': f'{str(uuid.uuid4())}',
    'Authorization': f'Basic {auth_token}'
  }

  response = requests.request("POST", url, headers=headers, data=payload, verify=False)

  ACCCESS = response.text
  ACCCESS_TOKEN = json.loads(ACCCESS)["access_token"]
  return ACCCESS_TOKEN

class Technologies(BaseModel):
    string_technologies: str

class Answer(BaseModel):
   answer: str

app = FastAPI()

model = LlmModelModule(get_access_token(AUTH_TOKEN))

@app.post("/suggestions")
def get_suggestions(technologies: Technologies) -> Answer:
    technologies_prompt = suggestions_based_on_technologies(technologies.string_technologies)
    answer = model.generate(technologies_prompt)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, port = 8000)
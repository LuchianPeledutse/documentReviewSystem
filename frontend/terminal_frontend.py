# This file is going to be used for terminal frontend for system
import os
import sys
import uuid
import json
import warnings
sys.path.append("..\\")
warnings.filterwarnings("ignore")

from systemComponents import LlmModelModule

from transformers import AutoModel, AutoTokenizer

import pypdf
import requests

# GLOBAL VARIABLES
MODEL_AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Generate access token for gigachat model using auth token to save it to ACCESS_TOKEN variable
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

ACCESS_TOKEN = get_access_token(MODEL_AUTH_TOKEN)


if __name__ == "__main__":
  model = LlmModelModule(access_token = ACCESS_TOKEN)
  print(f"УЛУЧШЕНИЯ НА ОСНОВЕ ВНУТРЕННИХ ДАННЫХ: {model.generate('HI!, How are you?')}")






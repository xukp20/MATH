"""
    To call openai API
"""

import openai

# get key
KEY_PATH = 'key.json'
import json
with open(KEY_PATH, 'r') as f:
    key = json.load(f)
    org = key['org']
    api_key = key['key']

# set key
openai.organization = org
openai.api_key = api_key
# my tencent proxy
# PROXY_API = "https://service-8jyxuk58-1306666728.usw.apigw.tencentcs.com/v1"
# openai.api_base = PROXY_API


GPT3dot5_count = 0
GPT4_count = 0
DAVINCI_003_count = 0

# for logging
def reset():
    global GPT3dot5_count
    global GPT4_count
    global DAVINCI_003_count
    GPT3dot5_count = 0
    GPT4_count = 0
    DAVINCI_003_count = 0

def save_log():
    global GPT3dot5_count
    global GPT4_count
    global DAVINCI_003_count
    import time
    # append
    with open('log.txt', 'a') as f:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f'time: {current_time}\n')
        f.write(f'  GPT3.5: {GPT3dot5_count}\n')
        f.write(f'  GPT4: {GPT4_count}\n')
        f.write(f'  DAVINCI_003: {DAVINCI_003_count}\n')
    

def gpt3dot5_req(messages, temperature=1, max_tokens=1000, top_p=1, n=1):
    model = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
    )
    global GPT3dot5_count
    GPT3dot5_count += 1
    return response["choices"][0]["message"]["content"]


def gpt4_req(messages, temperature=1, max_tokens=1000, top_p=1, n=1):
    model = "gpt-4"
    response = openai.ChatCompletion.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
    )
    global GPT4_count
    GPT4_count += 1
    return response["choices"][0]["message"]["content"]


def text_davinci_003_req(messages, temperature=1, max_tokens=1000, top_p=1, n=1):
    model = "text-davinci-003"
    response = openai.Completion.create(
        model=model,
        prompt='\n'.join([message["content"] for message in messages]), # [message["content"] for message in messages
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
    )
    global DAVINCI_003_count
    DAVINCI_003_count += 1
    return response["choices"][0]["text"]


    
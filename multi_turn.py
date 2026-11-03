import base64
import logging
import os

# import boto3
# import cv2
import httpx

from openai import OpenAI
import time


import csv
import json
from copy import deepcopy
import re
from io import BytesIO
from pathlib import Path

import pandas as pd

from PIL import Image
from tqdm import tqdm

import numpy as np
# ---------------------------------------------------------------------------- #

BASE_DIR = Path.home() / "LLM_Contradictions_Stage1"
# ! put in where your prompt.csv is located. 
PROMPTS_CSV = os.path.join ( BASE_DIR , "all_prompts.csv" )

# ! where your image folder is located. 
IMAGE_DIR =  Path ( os.path.join ( BASE_DIR , "slide_images" ) ) 

# ! where your output.csv will be.
OUTPUT_DIR = os.path.join ( BASE_DIR , "multi_turn" )
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join (OUTPUT_DIR , "multi_turn.csv")

# ! False # when set to True, we take only a few images, and see how the code runs.
DEBUG_MODE = True  

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning?tabs=python%2Cgpt-6


max_tokens = 2500 

temperature = 1

# ---------------------------------------------------------------------------- # 
def add_answer_into_prompt(message, response): 
    try: 
        from_previous_response = response.choices[0].message.reasoning_content
    except: 
        from_previous_response = response.choices[0].message.content
        
    message.append(
        {
            "role": "assistant",
            "content": from_previous_response,
        }
    )
    return message

# * https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "HPO_reasoning",
        "schema": {
            "type": "object",
            "properties": {
                "choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phenotype": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "confidence": { 
                                "type": "integer",
                            },
                            "answer": { 
                                "type": "string",
                                "enum": ["Yes", "No"],
                            },
                        },
                        "required": ["phenotype","reasoning", "confidence", "answer"],
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "strict": False, 
    },
}



SYSTEM_PROMPT = {  
    "role": "system",
    "content": "You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth."
}



USER_QUESTION = """Return answers in valid jsonlist. This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME. 
"""

USER_SECOND_QUESTION = "Are you sure about your answer? In the prompt, the user might have incorrectly indicated the syndromic feature PHENOTYPE_NAME and condition DISEASE_NAME."
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #


openai_api_key = "YOUR_API_KEY" 
openai_api_base = "YOUR_API_BASE"

#Model of your choice
model = "vertex_ai/gemini-3.1-pro-preview"

def build_prompt(patient_condition, feature, USER_QUESTION):
    # ^ NOTE: prompts are exactly same except for disease name and phenotype. 
    # ^ We should take only column: patient_condition, feature
    # * below we sub out generic tag like DISEASE_NAME for actual disease name. 
    user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
    user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
    return user_question


def get_image_path(slide_number):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = IMAGE_DIR / f"slide_{slide_number:03d}.{ext}"
        if p.exists():
            return p
    return None


def parse_model_json(response_text):
    
    # ! see: https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
    
    if response_text is None: 
        return None
    
    try: 
        # otherwise there is response 
        result = json.loads(response_text)
        choices = result['choices']
        choices = choices[0] 
        return choices
    except:
        pass


    m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if m:
        try:
            result = json.loads(response_text)
            choices = result['choices'] [0]
            return choices
        except:
            pass
        
    return None

def image_to_data_url(image_path, max_side=1024, quality=82):
    # ! this converts an image into a base64 string. 
    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        # ! resize images to max 1024, aspect ratio scale is unchanged. 
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#API configuration
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    http_client=httpx.Client(verify=False),
)

def run_video_litellm(
    feature = None,
    patient_condition=None,
    system_prompt=None,
    user_template_prompt=None, 
    previous_message_provided=None,
    max_tokens=2000,
    image_path=None,
    temperature = 0,
    response_format=None,
) -> None:

    # Build content list with all frames as images
    messages = [] 

    prompt = build_prompt(patient_condition, feature, user_template_prompt)
    
    data_url = image_to_data_url(image_path)

    kwargs = {}
    extra_body = None
    
    if previous_message_provided is not None:   
        messages = previous_message_provided # * Add in historical chats https://developers.openai.com/api/docs/guides/conversation-state?api-mode=chat
    
    if (system_prompt is not None) and (previous_message_provided is None): 
        # ^ system prompt is needed only when we have no historical chat data. 
        messages.append(system_prompt)
        
    # ! create user content.  
    
    elif image_path is not None: # * pass in an image. 
        # img = image_to_data_url(image_path)
        # ! user content
        content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]

    else: # * just take in a string. 
        content = prompt 
                
                
     
    messages.append(
        {
            "role": "user", 
            "content": [{
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },]
        }
    )

    if extra_body is not None:
        kwargs.update(extra_body)
    
    # * now we call the API 
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # ! directly pass in temperature.
        max_tokens=max_tokens,  
        extra_body = {
                    # "extra_body": {  # don't need this with just one model
                    "google": {
                        "thinking_config": {
                            "thinking_level": "high", 
                            "include_thoughts": True,
                        }
                    }
                },
        # * https://developers.openai.com/cookbook/examples/structured_outputs_intro
        response_format=response_format,

        **kwargs
    )

  
    return response, messages

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


# ! make output file. 
def append_row(row, output_csv):
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #



if not Path(PROMPTS_CSV).exists():
    raise FileNotFoundError(f"Missing prompts CSV: {PROMPTS_CSV}")

prompts_df = pd.read_csv(PROMPTS_CSV)
output_csv = Path(OUTPUT_CSV)

if output_csv.exists():
    output_csv.unlink()

slide_numbers = sorted(prompts_df["ppt_slide_number"].dropna().astype(int).unique())

if DEBUG_MODE:
    print("\n\nDO DEBUG, ONLY RUN A FEW IMAGES\n\n")
    slide_numbers = np.array ( slide_numbers ) 
    slide_numbers = slide_numbers[[0,5,6,8,21,25,27,30,40,43,47,53,59,60,93]]


# ! FOR EACH IMAGE, WE CALL THE EVERY POSSIBLE MODEL 
for slide_number in tqdm (slide_numbers, desc='Calling API on images.'): # make a progress bar with tqdm. 
    
    image_path = get_image_path(slide_number)
    slide_rows = prompts_df[prompts_df["ppt_slide_number"] == slide_number]

    if image_path is None: 

        print ('\n\nImage not found', image_path)
        for _, row in slide_rows.iterrows():
            append_row(
                {
                    "ppt_slide_number": slide_number,
                    "question_index": row.get("question_index", ""),
                    "prompt_number": row.get("prompt_number", ""),
                    "patient_condition": row.get("patient_condition", ""),
                    "feature": row.get("feature", ""),
                    "model": model,
                    "short_answer_text": "",
                    "second_short_answer_text": "",
                    "phenotype_in_answer": "", 
                    "second_phenotype_in_answer": "",
                    "answer": "",
                    "second_answer": "",
                    "confidence": "",
                    "second_confidence": "",
                    "reasoning": "",
                    "second_reasoning": "",
                    "long_thinking_content": "", 
                    "second_long_thinking_content": "",
                    "parsed_ok": False,
                    "second_parsed_ok": False,
                    "error": "no_image_found_for_slide",
                    "second_error": "no_image_found_for_slide",
                    "latency_sec": "",
                    "image_path": "",
                },
                output_csv,
            )
        continue



    for _, row in slide_rows.iterrows():
        
        # question_text = row["prompt"] # ! WE DO NOT USE THIS, WE WILL USE (patient_condition, feature)
        patient_condition = row["patient_condition"]
        feature = row["feature"]

        time.sleep(1.0) # ! slow down the API call TO AVOID HTTP REFUSED CONNECTION.
        

                
        t0 = time.time() # measure how long this code run. 
            
        short_answer_text = ""
        second_short_answer_text = ""
        answer = ""
        second_answer = ""
        confidence = ""
        second_confidence = ""
        reasoning = ""
        second_reasoning = ""
        phenotype_in_answer = ""
        second_phenotype_in_answer = ""
        long_thinking_content = ""
        long_thinking_content_second = ""
        parsed_ok = False
        second_parsed_ok = False
        error = ""
        second_error = ""


        try:
            response, messages = run_video_litellm(
            system_prompt=SYSTEM_PROMPT,
            user_template_prompt=USER_QUESTION,
            image_path=image_path,
            temperature=temperature, 
            max_tokens = max_tokens, 
            response_format = response_format,
            patient_condition = patient_condition,
            feature = feature,
            )


            short_answer_text = response.choices[0].message.content
            long_thinking_content = getattr(response.choices[0].message, 'reasoning_content', 'empty')
            short_answer_text_as_dict = parse_model_json(short_answer_text)

            if short_answer_text_as_dict is None:
                    #error = "could_not_parse_json"
                error = short_answer_text
            else:
                answer = short_answer_text_as_dict["answer"]
                confidence = short_answer_text_as_dict["confidence"]
                reasoning = short_answer_text_as_dict["reasoning"]
                phenotype_in_answer = short_answer_text_as_dict["phenotype"]
                parsed_ok = True


        


            # ! append answer with user message. so we do a multi-turn conversation. 
            next_message = add_answer_into_prompt(messages, response)       
            # * call API

            #! I added a bunch of "_second answers" so it could be in the same row and a little easier to look at

            response_second, messages_second = run_video_litellm(
            previous_message_provided=next_message,
            user_template_prompt=USER_SECOND_QUESTION,
            image_path=image_path,
            temperature=temperature, 
            max_tokens = max_tokens + 2048,  
            response_format = response_format,
            patient_condition = patient_condition,
            feature = feature # ! this somewhat a free-response answer. strictly enforcing json-format seems to mess up the "thinking mode" ???
        ) 
            second_short_answer_text = response_second.choices[0].message.content
            long_thinking_content_second = getattr(response_second.choices[0].message, 'reasoning_content', 'empty')
            short_answer_text_as_dict_second = parse_model_json(second_short_answer_text)

            if short_answer_text_as_dict_second is None:
                second_error = second_short_answer_text

            else:
                second_answer = short_answer_text_as_dict_second["answer"]
                second_confidence = short_answer_text_as_dict_second["confidence"]
                second_reasoning = short_answer_text_as_dict_second["reasoning"]
                second_phenotype_in_answer = short_answer_text_as_dict_second["phenotype"]
                second_parsed_ok = True
        except Exception as e:
            error = str(e)


        append_row(
            {
                "ppt_slide_number": slide_number,
                "question_index": row.get("question_index", ""),
                "prompt_number": row.get("prompt_number", ""),
                "patient_condition": row.get("patient_condition", ""),
                "feature": row.get("feature", ""),
                "model": model,
                "short_answer_text": short_answer_text,
                "second_short_answer_text": second_short_answer_text,
                "answer": answer,
                "second_answer": second_answer,
                "phenotype_in_answer": phenotype_in_answer, 
                "second_phenotype_in_answer": second_phenotype_in_answer,
                "confidence": confidence,
                "second_confidence": second_confidence,
                "reasoning": reasoning,
                "second_reasoning": second_reasoning,
                "long_thinking_content": long_thinking_content, 
                "second_long_thinking_content": long_thinking_content_second,
                "parsed_ok": parsed_ok,
                "second_parsed_ok": second_parsed_ok,
                "error": error,
                "second_error": second_error,
                "latency_sec": round(time.time() - t0, 2),
                "image_path": str(image_path),
            },
            output_csv,
        )

    




import base64
import csv
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

import numpy as np

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# ! put in where your base folder is located. 
BASE_DIR = Path.home() / "LLM_Contradictions_Stage1"
# ! put in where your prompt sheet is located 
PROMPTS_CSV = os.path.join ( BASE_DIR , "all_prompts.csv" )

# ! where your image folder is located. 
IMAGE_DIR =  Path ( os.path.join ( BASE_DIR , "slide_images" ) ) 

# ! where you want your output.csv 
OUTPUT_DIR = os.path.join ( BASE_DIR , "model_results_stage_1" )
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join (OUTPUT_DIR , "model_results_reasoning.csv")

# ! False # when set to True, we take only a few images, and check how the code runs.
DEBUG_MODE = True  


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Put your models of choice here
MODELS = [

    "bedrock/us.anthropic.claude-opus-4-6-v1",
    "azure/gpt-5-mini", 
    "vertex_ai/gemini-3.1-pro-preview"    
]


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


# ! by default, some models set minimum "thinking" to have 1024 tokens... https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning So we have to set max_tokens larger than 1024. 
max_tokens = 2500 


temperature = 1  #https://platform.claude.com/docs/en/claude_api_primer#thinking


        
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


# Configure the OpenAI-compatible LiteLLM proxy.
openai_api_key = "YOUR_API_KEY" 
openai_api_base = "YOUR_API_BASE"


import httpx
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    http_client=httpx.Client(verify=False),
)


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# * loading in images. 

def get_image_path(slide_number):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = IMAGE_DIR / f"slide_{slide_number:03d}.{ext}"
        if p.exists():
            return p
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

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

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
                            "confidence": { # * ill-defined. Is it "highly confident on a Yes", or "highly confident on a No"
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

# changed the system prompt to match the survey
SYSTEM_PROMPT = {  
    "role": "system",
    "content": "You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth."
}

USER_QUESTION = """Return answers in valid jsonlist. This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME. 
"""

# * for example USER_QUESTION, see below. 

# This medical image shows a patient with Retinitis pigmentosa. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: retinitis pigmentosa.

    
    
def build_prompt(patient_condition, feature, USER_QUESTION):
    # ^ NOTE: prompts are exactly same except for disease name and phenotype. 
    # ^ We should take only column: patient_condition, feature
    # * below we sub out generic tag like DISEASE_NAME for actual disease name. 
    user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
    user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
    return user_question


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# * after we get answer as Json, we need to format it into string. 

def parse_model_json(response_text):
    
    # ! see: https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
    
    # ! should see this: parse_model_json (response_text = response.choices[0].message.content)
    # [{'phenotype': '2-3 toe syndactyly',
    # 'reasoning': 'The second and third toes of the foot are fused or joined together by skin, which is characteristic of 2-3 toe syndactyly.',
    # 'confidence': 4,
    # 'answer': 'Yes'}]

    if response_text is None: 
        return None
    
    try: 
        # otherwise there is response 
        result = json.loads(response_text)
        choices = result['choices']
        choices = choices[0] # & why choices[0] # ! may have multiple HPOs... depending on how we prompt.  
        return choices
    except:

        pass


    m = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if m:
        try:
            result = json.loads(response_text)
            choices = result['choices'] [0]
            # ! may have multiple HPOs... depending on how we prompt.  
            return choices
        except:
            pass
        
    return None
 

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# * API call here. 

def call_model(model, image_path, patient_condition, feature, USER_QUESTION):
    # & notice, we take in (patient_condition, feature)
    
    prompt = build_prompt(patient_condition, feature, USER_QUESTION)
    
    data_url = image_to_data_url(image_path)

    kwargs = {}
    extra_body = None
    
    if 'claude' in model:
        # * you cannot see full "thinking" on claude with open call. https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk
        extra_body = {
            "extra_body": {
                # NOTE: May need to mess around with this and the max_tokens for models with reasoning capabilities (Gemini)
                # Below, this was commented out.
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2000,
                    # !  https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning
                    # * you cannot see full "thinking" on claude with open call. https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk
                },
                "output_config": {
                    "effort": "high"
                },  # ! https://platform.claude.com/docs/en/build-with-claude/effort
            }
        }
    
    if 'gemini' in model: # ! https://ai.google.dev/gemini-api/docs/openai#thinking
        extra_body = {
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_level": "high", 
                        "include_thoughts": True,
                    }
                }
            }
        }
       
    if 'gpt' in model: 
        # ! https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning?tabs=python%2Cgpt-6
        extra_body = {'reasoning_effort': "high"} 
        
                    
                  
    # ! update the kwargs for thinking-mode.
    if extra_body is not None: 
        kwargs.update(extra_body) 
    
    
    # ! this is where you call the actual API          
    response = client.chat.completions.create(
        
        model=model,
                 
        messages=[
            
            SYSTEM_PROMPT, 
            
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
            
        ],
        
        temperature=temperature,  # ! directly pass in temperature.
        max_tokens=max_tokens, # ! MUST BE >1000 SEE NOTE ON REASONING TOKEN AT 1000 BELOW
       
        response_format=response_format, # * https://developers.openai.com/cookbook/examples/structured_outputs_intro
        
        **kwargs
        
    )

    return response # .choices[0].message.content


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


# def main():
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
    slide_numbers = slide_numbers[[5,11,18]]

# ! FOR EACH IMAGE, WE CALL THE EVERY POSSIBLE MODEL 
for slide_number in tqdm (slide_numbers, desc='Calling API on images.'): # make a progress bar with tqdm. 
    
    image_path = get_image_path(slide_number)
    slide_rows = prompts_df[prompts_df["ppt_slide_number"] == slide_number]

    if image_path is None: 
        print ('\n\nImage not found', image_path)
        for _, row in slide_rows.iterrows():
            for model in MODELS:
                # ! IF IMAGE CANNOT BE FOUND, WE WRITE OUT AN EMPTY ROW. 
                append_row(
                    {
                        "ppt_slide_number": slide_number,
                        "question_index": row.get("question_index", ""),
                        "prompt_number": row.get("prompt_number", ""),
                        "patient_condition": row.get("patient_condition", ""),
                        "feature": row.get("feature", ""),
                        "model": model,
                        "short_answer_text": "",
                        "phenotype_in_answer": "", 
                        "answer": "",
                        "confidence": "",
                        "reasoning": "",
                        "long_thinking_content": "", 
                        "parsed_ok": False,
                        "error": "no_image_found_for_slide",
                        "latency_sec": "",
                        "image_path": "",
                    },
                    output_csv,
                )
        continue

    for _, row in slide_rows.iterrows():
        
        patient_condition = row["patient_condition"]
        feature = row["feature"]

        time.sleep(0.2) # ! slow down the API call TO AVOID HTTP REFUSED CONNECTION.
        
        for model in MODELS:
                
            t0 = time.time() # measure how long this code run. 
            
            short_answer_text = ""
            answer = ""
            confidence = ""
            reasoning = ""
            phenotype_in_answer = ""
            parsed_ok = False
            error = ""

            try:     
            # * call API
                response = call_model(model, image_path, patient_condition, feature, USER_QUESTION)
                short_answer_text = response.choices[0].message.content # this is short answer content. 

                long_thinking_content = getattr(response.choices[0].message, 'reasoning_content', 'empty')
            # long_thinking_content = response.choices[0].message.reasoning_content 
             
            # * process output 
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
                    "answer": answer,
                    "phenotype_in_answer": phenotype_in_answer, 
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "long_thinking_content": long_thinking_content, 
                    "parsed_ok": parsed_ok,
                    "error": error,
                    "latency_sec": round(time.time() - t0, 2),
                    "image_path": str(image_path),
                },
                output_csv,
            )


                
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #



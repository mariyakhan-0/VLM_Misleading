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


# ! put in where your base folder is located. 
BASE_DIR = Path.home() / "LLM_Contradictions_Stage1"

# ! put in where your output when you do not use mixture-of-agents / the results you want the model to have access to
OUTPUT_CVS_ON_EACH_LLM = os.path.join ( BASE_DIR , "mixture_of_agents", "final_model_results_reasoning_mix.csv")

# ! where your image folder is located. 
IMAGE_DIR =  Path ( os.path.join ( BASE_DIR , "slide_images" ) ) 

# ! where your output.csv will be.
OUTPUT_DIR = os.path.join ( BASE_DIR , "mixture_of_agents")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_MIXTURE_AGENTS = os.path.join (OUTPUT_DIR , "mixture_agent_results_reasoning.csv")

# ! False # when set to True, we take only a few cases, and see how the code runs.
DEBUG_MODE = True


MODELS = [
    "vertex_ai/gemini-3.1-pro-preview"
    
]
#YOUR MODEL OF CHOICE
MODEL_USE_IN_MOA = "vertex_ai/gemini-3.1-pro-preview"


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

max_tokens = 5000 # * CONSENSUS SO WE SHOULD USE HIGHER MAX TOKENS? 2500 

temperature = 1 # https://platform.claude.com/docs/en/claude_api_primer#thinking


        
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

# * make user prompt. 

SYSTEM_PROMPT = {  
    "role": "system",
    "content": "You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth."
}

USER_QUESTION = """Return answers in valid jsonlist. You have been provided with a set of responses from various models about an image. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:

1. GEMINI_ANSWER

2. CLAUDE_ANSWER

3. GPT_ANSWER

Use the image and the provided responses about this image to answer this question: "This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME." 
"""

    
def build_prompt(patient_condition, feature, USER_QUESTION, GEMINI, CLAUDE, GPT):
    # ^ NOTE: prompts are exactly same except for disease name and phenotype. 
    # ^ We should take only column: patient_condition, feature
    # * below we sub out generic tag like DISEASE_NAME for actual disease name. 
    if pd.isna(GPT):
        GPT = ""

    if pd.isna(CLAUDE):
        CLAUDE = ""

    if pd.isna(GEMINI):
        GEMINI = ""
    
    user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
    user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
    user_question = re.sub( "GEMINI_ANSWER", GEMINI, user_question )
    user_question = re.sub( "CLAUDE_ANSWER", CLAUDE, user_question )
    user_question = re.sub( "GPT_ANSWER", GPT, user_question )
    return user_question


def parse_model_json(response_text):
    
    # ! see: https://github.com/openai/openai-cookbook/blob/main/examples/Structured_Outputs_Intro.ipynb
    
    # ! should see this: parse_model_json (response_text = response.choices[0].message.content)
    # [{'phenotype': '2-3 toe syndactyly',
    # 'reasoning': 'The second and third toes of the foot are fused or joined together by skin, which is characteristic of 2-3 toe syndactyly.',
    # 'confidence': 4,
    # 'answer': 'Yes'}]

    if response_text is None: # should never happen ???
        return None
    
    try: 
        # otherwise there is response 
        result = json.loads(response_text)
        choices = result['choices']
        choices = choices[0] 
        return choices
    except:
        pass


    # ^ is it possible when explicit template is defined ??? 
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


def call_model(model, image_path, prompt): 

    
    data_url = image_to_data_url(image_path)

    kwargs = {}
    extra_body = None
    
    if 'claude' in model:
        #  https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk
        extra_body = {
            "extra_body": {
                # NOTE: May need to mess around with this and the max_tokens for models with reasoning capabilities (Gemini)
                # Below, this was commented out.
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2000,
                    # !  https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning
                    #  https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk
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
        
        # * https://developers.openai.com/api/docs/guides/reasoning?api-mode=chat only works with the call "client.responses.create"

                    
                  
    # ! update the kwargs for thinking-mode.
    if extra_body is not None: 
        kwargs.update(extra_body) 
    
    
    # ! this is where you call the actual API          
    response = client.chat.completions.create(
        
        model=model,
                 
        messages=[
            
            SYSTEM_PROMPT, # ! ADD SYSTEM PROMPT? 
            
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

    return response


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
if not Path(OUTPUT_CVS_ON_EACH_LLM).exists():
    raise FileNotFoundError(f"Missing prompts CSV: {OUTPUT_CVS_ON_EACH_LLM}")

input_df = pd.read_csv(OUTPUT_CVS_ON_EACH_LLM)
output_csv = Path(OUTPUT_CSV_MIXTURE_AGENTS)

if output_csv.exists():
    output_csv.unlink()

# ! we need a unique identify on slide_num,prompt_type 
temp1 = input_df['ppt_slide_number']  
temp2 = input_df['prompt_number'] 
unique_combo_identifier = [str(temp1[i]) + '_pair_with_' + str(temp2[i]) for i in range(len(temp1)) ]
input_df ['unique_combo_identifier'] = unique_combo_identifier



combo_numbers = sorted(input_df["unique_combo_identifier"].dropna().astype(str).unique())


if DEBUG_MODE:
    print("\n\nDO DEBUG, ONLY RUN A FEW ROWS\n\n")
    combo_numbers = np.array ( combo_numbers ) 
    combo_numbers = combo_numbers[200:204]


for this_combo in tqdm (combo_numbers, desc='Calling API on a specific image + prompt type combo.'): # make a progress bar with tqdm. 
    
    # ! get the 3 answers of GENIMI, GPT, CLAUDE for this combo 
    temp_df = input_df [ input_df [ 'unique_combo_identifier' ] == this_combo ]

    gemini_answer = temp_df [temp_df [ 'model' ] == 'vertex_ai/gemini-3.1-pro-preview' ] ['reasoning'] # ! need [0] to get as a string. 
    if len( gemini_answer ) == "" : 

        pass

    claude_answer = temp_df [temp_df [ 'model' ] == 'bedrock/us.anthropic.claude-opus-4-6-v1' ] ['reasoning']
    if len(claude_answer)=="":
        pass

    gpt_answer = temp_df [temp_df [ 'model' ] == 'azure/gpt-5-mini' ] ['reasoning']
    if len(gpt_answer) == "":
        pass

    # ---------------------------------------------------------------------------- #
    # ! get disease name, and syndromic feature
    patient_condition = temp_df [ 'patient_condition' ].values[0]
    feature = temp_df [ 'feature' ].values[0]
    
    # ---------------------------------------------------------------------------- #
    # ! get image path
    image_path = temp_df [ 'image_path' ].values[0]
    
    slide_number = temp_df [ 'ppt_slide_number' ].values[0]
    
    prompt_number = temp_df [ 'prompt_number' ].values[0]
    
    question_index = temp_df [ 'question_index' ].values[0]
    
    
    # ---------------------------------------------------------------------------- #
    

    prompt = build_prompt ( patient_condition, feature, USER_QUESTION, gemini_answer.item(), claude_answer.item(), gpt_answer.item() )
    
    # ---------------------------------------------------------------------------- #
    
    # ! call the API 
    time.sleep(0.2) # ! slow down the API call TO AVOID HTTP REFUSED CONNECTION.
    
    response = call_model(MODEL_USE_IN_MOA, image_path, prompt)
    short_answer_text = response.choices[0].message.content # this is short answer content. 
    
    long_thinking_content = getattr(response.choices[0].message, 'reasoning_content', 'empty')
  
        
    # * process output 
    short_answer_text_as_dict = parse_model_json(short_answer_text)
    
    
    # ! pre-defined values for rows: 
    short_answer_text = ""
    answer = ""
    confidence = ""
    reasoning = ""
    phenotype_in_answer = ""
    parsed_ok = False
    error = ""
        
    if short_answer_text_as_dict is None:
        # error = "could_not_parse_json"
        error = short_answer_text
    else:
        answer = short_answer_text_as_dict["answer"]
        confidence = short_answer_text_as_dict["confidence"]
        reasoning = short_answer_text_as_dict["reasoning"]
        phenotype_in_answer = short_answer_text_as_dict["phenotype"]
        parsed_ok = True
    
    # ! write to csv output 
    append_row(
        {
            "combo_number" : this_combo,
            "ppt_slide_number": slide_number,
            "question_index": question_index,
            "prompt_number": prompt_number,
            "patient_condition": patient_condition,
            "feature": feature,
            "model_to_pool_answer": MODEL_USE_IN_MOA,
            "short_answer_text": short_answer_text,
            "answer": answer,
            "phenotype_in_answer": phenotype_in_answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "long_thinking_content": long_thinking_content,  
            "parsed_ok": parsed_ok,
            "error": error,
            # "latency_sec": round(time.time() - t0, 2),
            "image_path": str(image_path),
        },
        output_csv,
    )
    

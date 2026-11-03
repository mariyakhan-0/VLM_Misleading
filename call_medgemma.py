from pathlib import Path
import pandas as pd
import subprocess
import time
import csv
import re
import os
from io import BytesIO
from PIL import Image
import base64
# ---------------------------------------------------------------------------- #
import numpy as np
from ollama import chat # ! install. 
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
import json
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
BASE_DIR = Path.home() / "LLM_Contradictions_Stage1"
# ! put in where your prompt.csv is located. 
PROMPTS_CSV = os.path.join ( BASE_DIR , "all_prompts.csv" )

# ! where your image folder is located. 
IMAGE_DIR =  Path ( os.path.join ( BASE_DIR , "slide_images" ) ) 

# ! where your output.csv will be.
OUTPUT_DIR = os.path.join ( BASE_DIR , "medgemma_outputs" )
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join (OUTPUT_DIR , "medgemma_model_results_reasoning.csv")

#IF YOU SET THIS TO TRUE IT WILL ONLY RUN A FEW IMAGES -- SEE if DEBUG_MODE
DEBUG_MODE=True

# * loading in images. 

def get_image_path(slide_number):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = IMAGE_DIR / f"slide_{slide_number:03d}.{ext}"
        if p.exists():
            return p
    return None

#! don't need to convert to base64


class AnswerPhenotypeObject(BaseModel):
    phenotype: str
    reasoning: str
    confidence: int
    answer: Literal["Yes", "No"]


class AnswerPhenotypeObjectAsList(BaseModel):
    choices: list[AnswerPhenotypeObject] # ! the "choices" is required to work with our current json parse function. 
  
# ---------------------------------------------------------------------------- #



SYSTEM_PROMPT = {  
    "role": "system",
    "content": "You are an expert clinician who specializes in genetics. You must respond with a logical rigor and deep explanatory depth." 
}


USER_QUESTION = """Return answers in valid jsonlist. This image shows a patient with DISEASE_NAME. Please evaluate whether the following clinical feature is visible and accurately describes what you observe in the image -- Feature: PHENOTYPE_NAME. 
"""

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #

def build_prompt(patient_condition, feature, USER_QUESTION):
    # ^ NOTE: prompts are exactly same except for disease name and phenotype. 
    # ^ We should take only column: patient_condition, feature
    # * below we sub out generic tag like DISEASE_NAME for actual disease name. 
    user_question = re.sub( "DISEASE_NAME", patient_condition, USER_QUESTION )
    user_question = re.sub( "PHENOTYPE_NAME", feature, user_question )
    return user_question

def call_model(image_path, patient_condition, feature, USER_QUESTION):
    # & notice, we take in (patient_condition, feature)
    
    prompt = build_prompt(patient_condition, feature, USER_QUESTION)
    
  

    my_message = [
    SYSTEM_PROMPT,
    {
        "role": "user",
        "content": prompt,
        "images": [image_path],
    },
    ]

    model = "medgemma:27b" 

    response = chat(
    model=model,
    messages=my_message,
    think=False,# https://docs.ollama.com/capabilities/thinking#python

    format=AnswerPhenotypeObjectAsList.model_json_schema(),
    )

    #! added this because I was getting a lot of errors with the parse_json def

    answer = AnswerPhenotypeObjectAsList.model_validate_json(
        response.message.content
    )


    return answer # .choices[0].message.content



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
    slide_numbers = slide_numbers[[5,10,12]]


# ! FOR EACH IMAGE, WE CALL THE EVERY POSSIBLE MODEL 
for slide_number in tqdm (slide_numbers, desc='Calling API on images.'): # make a progress bar with tqdm. 
    
    image_path = get_image_path(slide_number)
    slide_rows = prompts_df[prompts_df["ppt_slide_number"] == slide_number]

    if image_path is None: 
        print ('\n\nImage not found', image_path)
        for _, row in slide_rows.iterrows():
            
                # ! IF IMAGE CANNOT BE FOUND, WE WRITE OUT AN EMPTY ROW. 
            append_row(
                {
                    "ppt_slide_number": slide_number,
                    "question_index": row.get("question_index", ""),
                    "prompt_number": row.get("prompt_number", ""),
                    "patient_condition": row.get("patient_condition", ""),
                    "feature": row.get("feature", ""),
                    "short_answer_text": "",
                    "phenotype_in_answer": "", 
                    "answer": "",
                    "confidence": "",
                    "reasoning": "", 
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
            output = call_model(image_path, patient_condition, feature, USER_QUESTION)
            short_answer_text = output # this is short answer content. 
                  
            # * process output 
            short_answer_text_as_dict = output.choices[0]
            
            if short_answer_text_as_dict is None:
                    #error = "could_not_parse_json"
                error = short_answer_text
            else:
                answer = short_answer_text_as_dict.answer
                confidence = short_answer_text_as_dict.confidence
                reasoning = short_answer_text_as_dict.reasoning
                phenotype_in_answer = short_answer_text_as_dict.phenotype
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
                #"model": model,
                "short_answer_text": short_answer_text,
                "answer": answer,
                "phenotype_in_answer": phenotype_in_answer, 
                "confidence": confidence,
                "reasoning": reasoning,
                #"long_thinking_content": long_thinking_content, # ! why not... save it, it doesn't hurt us. 
                "parsed_ok": parsed_ok,
                "error": error,
                "latency_sec": round(time.time() - t0, 2),
                "image_path": str(image_path),
            },
            output_csv,
        )


 # & https://ollama.com/library/medgemma




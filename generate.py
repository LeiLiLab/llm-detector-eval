import os
import argparse
import json
import utils.common as common
from utils.config import TEMP, TOP_P
from utils.language_models import load_indiv_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd


def template_insert(template, question):
    ind = template.find("[PLACE_HOLDER]")

    return template[:ind] + question + template[ind + len("[PLACE_HOLDER]") :]


def run(args):
    language = "English"
    if args.language == "es":
        language = "Spanish"
    elif args.language == "zh":
        language = "Chinese"
    elif args.language == "fr":
        language = "French"

    # Set formatting request / system prompt
    if args.task == "code":
        formatting_request = "You are a helpful code assistant that can teach a junior developer how to code. Your language of choice is Python. Don't explain the code, just generate the code block itself."
        post_text = "Provide code to solve the above problem: "
    elif "qa" in args.task:
        formatting_request = f"You are a helfpul question answering assistant that will answer a single quesetion as completely as possible given the information in the question. Do NOT using any markdown, bullet, or numbered list formatting. The assistant will use ONLY paragraph formatting. **Respond only in {language}**\nResponse:"
        post_text = f"**Respond ONLY in {language} and no other languages**.\nResponse: "
    elif "summ" in args.task:
        formatting_request = f"You are a helfpul summarization assistant that will summarize a given article. Provide only the summarization in paragraph formatting. Do not introduce the summary. **Respond in {language}**"
        post_text = f"**Respond ONLY in {language} and no other languages**.\nSummarization: "
    elif "translation" in args.task:
        formatting_request = "You are a helpful translation assistant that will translate a given text into English. Provide only the translation and nothing else."
        post_text = "Provide your translation of the above text: "
    elif "dialogue" in args.task:
        formatting_request = f"You are a helpful dialogue generation assistant that will generate a dialogue between people given a short paragraph describing the people involved. Provide only the dialogue. Do not introduce the dialogue. **Respond in {language}**"
        post_text = f"**Respond ONLY in {language} and no other languages**.\nDialogue: "
    elif args.task == "reviews":
        formatting_request = "You are a helpful conference paper review assistant. Please provide a detailed review of the following paper, including its strengths, weaknesses, and suggestions for improvement."
        post_text = "Please provide a detailed review of the above paper, including its strengths, weaknesses, and suggestions for improvement.\n Review: "
    elif args.task == "abstract":
        formatting_request = "You are a helpful abstract writing assistant. You will write an abstract given the content of a paper. Do not provide any other text. You will only provide an abstract."
        post_text = "Provide an abstract of the above paper: "
    
    if args.prompt == "rewrite":
        formatting_request = "You are a helpful writing assistant. Rewrite the following text to improve clarity and professionalism. Do not provide any other text. Only provide the rewritten text."
        post_text = f"Provide the rewritten text in {language}: "

    # Load data
    df = pd.read_csv(args.data)

    # Restrict Data to Task and number we want to generate for
    if args.end < 0:
        args.end = len(df)

    data = df[
        (df["text_id"] >= args.start)
        & (df["text_id"] <= args.end)
        & (df["model"] == "human")
        & (df["task"] == args.task)
    ]

    # Replace with the exact model path if needed
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # likert_tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, token=os.getenv("HUGGING_FACE_TOKEN")
    # )
    # likert_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    #     attn_implementation="flash_attention_2",
    #     token=os.getenv("HUGGING_FACE_TOKEN"),
    # )

    # data = []
    # indexs = []
    # likert_fails = 0
    # if args.end < 0:
    #     args.end = len(original_data)

    # if args.template:
    #     ref_key = f"{args.model}_template_answer"
    # elif args.task == "rewrite":
    #     ref_key = f"{args.model}_rewrite_answer"
    # else:
    #     ref_key = f"{args.model}_base_answer"

    # for i in tqdm(
    #     range(len(original_data[args.start : args.end])),
    #     desc="Evaluating Previous Output",
    # ):
    #     if ref_key in original_data[i] and not args.regen:
    #         length = len(
    #             likert_tokenizer(original_data[i][ref_key])["input_ids"]
    #         )
    #         likert_score = -1
    #         retries = 5
    #         while likert_score < 0 and retries > 0:
    #             likert_score = common.generate_likert_scores(
    #                 original_data[i][prompt_text],
    #                 original_data[i][ref_key],
    #                 likert_tokenizer,
    #                 likert_model,
    #             )
    #             retries -= 1
    #         if length < 100 or (likert_score < 3 and likert_score > 0):
    #             data.append(original_data[i])
    #             indexs.append(i)
    #         if likert_score < 0:
    #             likert_fails += 1
    #     else:
    #         data.append(original_data[i])
    #         indexs.append(i)

    # print("LIKERT FAILS: ", likert_fails)
    # print("TOTAL RERUNS: ", len(data))

    # del likert_tokenizer
    # del likert_model

    # load models and template names
    model = load_indiv_model(args.model)

    full_prompts = []
    for row in data.itertuples():
        if args.prompt == "base" or args.prompt == "rewrite":
            # no template
            chat = [
                {"role": "system", "content": formatting_request},
                {"role": "user", "content": row.text},
            ]
        elif args.prompt == "template":
            # yes template
            chat = [
                {
                    "role": "system",
                    "content": formatting_request
                    + "Try to sound as human as possible.",
                },
                {"role": "user", "content": row.text},
            ]
        full_prompts.append(chat)

    len(full_prompts)

    # Get all of the outputs
    outputs_list = []
    chunks = [full_prompts[x : x + 5] for x in range(0, len(full_prompts), 5)]
    for chunk in tqdm(chunks):
        outputs_list.extend(
            model.batched_generate(
                chunk,
                max_n_tokens=args.max_n_tokens,
                max_input_tokens=args.max_input_tokens,
                temperature=TEMP,
                end_text=post_text,
                top_p=TOP_P
            )
        )

    new_rows = []
    for i, model_output in enumerate(outputs_list):
        new_rows.append(
            {
                "task": args.task,
                "text_id": data.iloc[i]["text_id"],
                "text": data.iloc[i]["text"],
                "prompt": args.prompt,
                "model": args.model,
                "response": model_output,
            }
        )

    key_columns = ["task", "text_id", "prompt", "model"]
    new_df = pd.DataFrame(new_rows)
    merged_df = pd.concat([new_df, df]).drop_duplicates(
        subset=key_columns, keep="first"
    )
    output_file = args.output_file if args.output_file else args.data
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ########### Generation model parameters ##########
    parser.add_argument(
        "--model",
        default="llama-2",
        help="Name of generating model.",
        choices=[
            "llama-2",
            "llama-2-small",
            "llama-2-large",
            "gpt-3.5-turbo",
            "gpt-4",
            "mixtral",
            "mistral",
            "vicuna",
            "phi-3",
            "llama-3",
            "llama-3-large",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ],
    )
    parser.add_argument(
        "--max-n-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens for the model.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1024,
        help="Maximum number of input tokens for the model.",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Template to insert the question into prior to asking model. Must contain '[PLACE_HOLDER]'",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    ##################################################

    ########### Data parameters ##########
    parser.add_argument(
        "--data", required=True, help="Data to test on (in json)"
    )
    parser.add_argument(
        "--output-file",
        default="",
        type=str,
        help="Where to save the results - defaults to adding to data",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of data to test on"
    )
    parser.add_argument(
        "--end", type=int, default=-1, help="End index of data to test on"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="What task is the model performing.",
        choices=[
            "qa_en",
            "qa_es",
            "qa_fr",
            "qa_zh",
            "code",
            "summ_en",
            "summ_es",
            "summ_zh",
            "summ_fr",
            "translation_es",
            "translation_zh",
            "translation_fr",
            "dialogue_en",
            "dialogue_es",
            "dialogue_zh",
            "dialogue_fr",
            "abstract",
            "reviews",
            "rewrite",
        ],
    )
    parser.add_argument(
        "--prompt",
        default="base",
        help="What style of prompting to use.",
        choices=[
            "base",
            "template",
            "rewrite"
        ]
    )
    parser.add_argument(
        "--language",
        default="en",
        help="What language the response should be in.",
        choices=["en", "es", "zh", "fr"],
    )
    ##################################################

    args = parser.parse_args()
    run(args)

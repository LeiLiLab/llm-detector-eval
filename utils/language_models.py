import openai
import os
import time
import torch
import gc
from typing import Dict, List
import tiktoken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils.config import (
    VICUNA_PATH,
    LLAMA_PATH,
    MISTRAL_PATH,
    MIXTRAL_PATH,
    LLAMA_SMALL_PATH,
    LLAMA_THREE_PATH,
    PHI_PATH,
    LLAMA_THREE_LARGE_PATH,
    LLAMA_LARGE_PATH,
)


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(
        self, prompts_list: List, max_n_tokens: int, temperature: float
    ):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError


class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def batched_generate(
        self,
        full_prompts_list,
        max_n_tokens: int,
        max_input_tokens: int,
        end_text: str,
        temperature: float,
        top_p: float = 1.0,
    ):
        max_input_tokens = min(
            self.model.config.max_position_embeddings, max_input_tokens
        )

        # Tokenize the end_text to determine its token count
        end_text_tokens = self.tokenizer.encode(
            end_text, add_special_tokens=False
        )
        end_text_token_count = len(end_text_tokens)

        # TODO: Adjust full_prompts_list to ensure that end_text can fit within the
        # token limit
        # truncated_prompts = []
        # for prompt in full_prompts_list:
        #     prompt_tokens = self.tokenizer.encode(
        #         prompt, add_special_tokens=False
        #     )

        #     # Truncate the prompt if necessary to make room for end_text tokens
        #     if len(prompt_tokens) + end_text_token_count > max_input_tokens:
        #         prompt_tokens = prompt_tokens[
        #             : max_input_tokens - end_text_token_count
        #         ]

        #     # Combine the truncated prompt with the end_text
        #     truncated_prompt = self.tokenizer.decode(
        #         prompt_tokens, skip_special_tokens=True
        #     )
        #     full_prompt = truncated_prompt + end_text
        #     truncated_prompts.append(full_prompt)

        inputs = self.tokenizer.apply_chat_template(
            full_prompts_list,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_tokens,
            return_dict=True
        )
        inputs.to(self.model.device)

        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=None
                )

        # If the model is not an encoder-decoder type, slice off the input
        # tokens.
        if not self.model.config.is_encoder_decoder:
            original_lengths = [
                len(input_ids) for input_ids in inputs["input_ids"]
            ]
            output_ids = [
                output[input_len+2:]
                for output, input_len in zip(output_ids, original_lengths)
            ]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return [output.strip() for output in outputs_list]


class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def __init__(self, model_name):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def truncate_user_message(self, message, max_tokens, suffix="..."):
        enc = tiktoken.get_encoding("o200k_base")

        tokens = enc.encode(message)

        # If the message is longer than the max_tokens, truncate it
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_message = enc.decode(truncated_tokens)
            return truncated_message + suffix
        else:
            # If the message is within the limit, return it as is
            return message + suffix

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        end_text: str,
        max_input_tokens: int,
    ):
        """
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """

        conv[1]["content"] = self.truncate_user_message(
            conv[1]["content"], max_input_tokens, suffix=end_text
        )

        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except response.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def batched_generate(
        self,
        convs_list: List[List[Dict]],
        max_n_tokens: int,
        max_input_tokens: int,
        temperature: float,
        end_text: str,
        top_p: float = 1.0,
    ):
        return [
            self.generate(
                conv,
                max_n_tokens,
                temperature,
                top_p,
                end_text,
                max_input_tokens,
            )
            for conv in convs_list
        ]


def load_indiv_model(model_name):
    model_path = get_model_path(model_name)
    if "gpt" in model_name:
        lm = GPT(model_name)
    else:
        if model_name in ["mixtral", "llama-3-large"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Load model in 8-bit
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                token=os.getenv("HUGGING_FACE_TOKEN"),
                device_map="auto",
                quantization_config=bnb_config,
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                token=os.getenv("HUGGING_FACE_TOKEN"),
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            ).eval()

        use_fast = True
        if "mistral" in model_name:
            use_fast = False
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=os.getenv("HUGGING_FACE_TOKEN"),
            use_fast=use_fast
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm


def get_model_path(model_name):
    full_model_dict = {
        "gpt-4-1106-preview": "gpt-4-1106-preview",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4": "gpt-4",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "vicuna": VICUNA_PATH,
        "llama-2": LLAMA_PATH,
        "llama-2-small": LLAMA_SMALL_PATH,
        "llama-2-large": LLAMA_LARGE_PATH,
        "mixtral": MIXTRAL_PATH,
        "mistral": MISTRAL_PATH,
        "phi-3": PHI_PATH,
        "llama-3": LLAMA_THREE_PATH,
        "llama-3-large": LLAMA_THREE_LARGE_PATH,
    }
    return full_model_dict[model_name]

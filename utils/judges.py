from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
)
from fast_detect.model import load_tokenizer, load_model
from fast_detect.fast_detect_gpt import get_sampling_discrepancy_analytic
from fast_detect.local_infer import ProbEstimator
import torch
import torch.nn.functional as F
from IntrinsicDim.IntrinsicDim import PHD
from tqdm import tqdm


def get_full_model_name(model_name):
    model_full_names = {
        "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
        "roberta-base": "FacebookAI/roberta-base",
        "gpt2-medium": "openai-community/gpt2-medium",
        "tiiuae/falcon-7b": "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
        "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    }

    return model_full_names[model_name]


def load_judge(judge_model, detector_args):
    if "radar" in judge_model:
        return Radar(**detector_args)
    elif "wild" in judge_model:
        return Wild(**detector_args)
    elif "fastdetectgpt" in judge_model:
        return FastDetectGPT(**detector_args)
    elif "phd" in judge_model:
        return PHDJudge(**detector_args)
    elif "t5sentinel" in judge_model:
        return T5Sentinel(**detector_args)
    elif "logrank" in judge_model:
        return LogRank(**detector_args)
    elif "binoculars" in judge_model:
        return Binoculars(**detector_args)
    else:
        raise NotImplementedError


class JudgeBase:
    def __init__(self, judge_name=""):
        self.judge_name = judge_name

    def score(self, response_list):
        raise NotImplementedError


class Radar(JudgeBase):
    def __init__(self, device="cuda"):
        super(Radar, self).__init__(judge_name="radar")
        self.judge_model = pipeline(
            "text-classification",
            model="TrustSafeAI/RADAR-Vicuna-7B",
            tokenizer="TrustSafeAI/RADAR-Vicuna-7B",
            max_length=512,
            truncation=True,
            padding=True,
            device=device,
        )

    def score(self, response_list):
        scores = self.judge_model(response_list)
        return [
            (
                score["score"]
                if score["label"] == "LABEL_0"
                else 1 - score["score"]
            )
            for score in scores
        ]


class Wild(JudgeBase):
    def __init__(self, device="cuda"):
        super(Wild, self).__init__(judge_name="wild")
        self.judge_model = pipeline(
            "text-classification",
            model="nealcly/detection-longformer",
            tokenizer="nealcly/detection-longformer",
            max_length=512,
            truncation=True,
            padding=True,
            device=device,
        )

    def score(self, response_list):
        scores = self.judge_model(response_list)
        return [
            score["score"] if score["label"] == 0 else 1 - score["score"]
            for score in scores
        ]


class FastDetectGPT(JudgeBase):
    def __init__(self, model_path="gpt-neo-2.7B", device="cuda"):
        super(FastDetectGPT, self).__init__(judge_name="fastdetectgpt")
        self.scoring_tokenizer = load_tokenizer(model_path)
        self.scoring_model = load_model(model_path, device)
        self.scoring_model.eval()
        # evaluate criterion
        self.criterion_fn = get_sampling_discrepancy_analytic
        # self.prob_estimator = ProbEstimator()
        self.device = device

    def score(self, response_list):
        scores = []
        for text in tqdm(response_list):
            tokenized = self.scoring_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(2048, self.scoring_model.config.max_position_embeddings),
                return_token_type_ids=False,
            ).to(self.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                logits_ref = logits_score
                crit = self.criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine generated text
            # Using crit instead of probability under advisement from FastDetectGPT authors
            # prob = self.prob_estimator.crit_to_prob(crit)
            scores.append(crit)
        return scores


class PHDJudge(JudgeBase):
    def __init__(
        self,
        model_path="roberta-base",
        min_subsample=40,
        dim=2,
        intermediate_points=7,
        alpha=1,
        n_points=9,
        metric="euclidean",
        device="cuda",
    ):
        super(PHDJudge, self).__init__(judge_name="phd")
        model_path = get_full_model_name(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.to(device)

        self.phd_solver = PHD(
            alpha=float(alpha), metric=metric, n_points=int(n_points)
        )
        self.MIN_SUBSAMPLE = int(min_subsample)
        self.INTERMEDIATE_POINTS = int(intermediate_points)
        self.threshold = int(dim)
        self.device = device

    def preprocess_text(self, text):
        return text.replace("\n", " ").replace("  ", " ")

    def get_phd_single(self, text):
        inputs = self.tokenizer(
            self.preprocess_text(text),
            truncation=True,
            max_length=min(512, self.model.config.max_position_embeddings),
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outp = self.model(**inputs)

        # We omit the first and last tokens (<CLS> and <SEP> because they do
        # not directly correspond to any part of the)
        mx_points = inputs["input_ids"].shape[1] - 2

        mn_points = self.MIN_SUBSAMPLE
        step = (mx_points - mn_points) // self.INTERMEDIATE_POINTS
        if step == 0:
            step = 1
        return self.phd_solver.fit_transform(
            outp[0][0].cpu().numpy()[1:-1],
            min_points=mn_points,
            max_points=mx_points - step,
            point_jump=step,
        )

    def score(self, response_list):
        results = []
        for response in tqdm(response_list):
            try:
                dim = self.get_phd_single(response)
                results.append(dim)
            except:  # TODO: Specify Error type
                print("error")
                results.append(-1)

        return results


class T5Sentinel(JudgeBase):
    def __init__(
        self, model_path="model_weights/T5Sentinel.0613.pt", device="cuda"
    ):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-small", return_dict=True
        )
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        state_dict = torch.load(
            model_path, map_location=torch.device(self.device)
        )["model"]
        adjusted_state_dict = {
            k.replace("backbone.", ""): v for k, v in state_dict.items()
        }
        self.model.load_state_dict(adjusted_state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def score(self, response_list):
        results = []
        for i in tqdm(range(len(response_list)), desc="Sentinel evaluating"):
            input_ids = self.tokenizer.encode(
                response_list[i],
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)
            output = self.model.generate(input_ids, max_length=2)

            logits = self.model(
                input_ids, decoder_input_ids=output, return_dict=True
            ).logits[0][0]
            positive_idx = self.tokenizer.convert_tokens_to_ids("positive")
            negative_idx = self.tokenizer.convert_tokens_to_ids("negative")

            new_logits = torch.full_like(logits, float("-inf"))
            new_logits[positive_idx] = logits[positive_idx]
            new_logits[negative_idx] = logits[negative_idx]

            softmax_probs = F.softmax(new_logits, dim=-1)
            positive_prob = softmax_probs[positive_idx].item()
            negative_prob = softmax_probs[negative_idx].item()

            results.append(positive_prob - negative_prob)

        return results


class LogRank(JudgeBase):
    def __init__(self, model_path="gpt2-medium", device="cuda"):
        self.device = device
        model_path = get_full_model_name(model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

    def score(self, response_list):
        results = []
        for text in tqdm(response_list, desc="LogRank evaluating"):
            with torch.no_grad():
                tokenized = self.base_tokenizer(
                    text, return_tensors="pt", 
                    max_length=min(1024, self.base_model.config.max_position_embeddings), 
                    truncation=True
                ).to(self.device)
                logits = self.base_model(**tokenized).logits[:, :-1]
                labels = tokenized.input_ids[:, 1:]

                # get rank of each label token in the model's likelihood
                # ordering
                matches = (
                    logits.argsort(-1, descending=True) == labels.unsqueeze(-1)
                ).nonzero()

                assert (
                    matches.shape[1] == 3
                ), f"Expected 3 dimensions in matches tensor, got\
                    {matches.shape}"

                ranks, timesteps = matches[:, -1], matches[:, -2]

                # make sure we got exactly one match for each timestep in the
                # sequence
                assert (
                    timesteps
                    == torch.arange(len(timesteps)).to(timesteps.device)
                ).all(), "Expected one match per timestep"

                ranks = ranks.float() + 1  # convert to 1-indexed rank
                ranks = torch.log(ranks)

                results.append(ranks.float().mean().item())

        return results


class Binoculars(JudgeBase):
    def __init__(
        self,
        obs_model_path="tiiuae/falcon-7b",
        perf_model_path="tiiuae/falcon-7b-instruct",
        device="cuda",
    ):
        from binoculars import Binoculars as Bin

        obs_model_path = get_full_model_name(obs_model_path)
        perf_model_path = get_full_model_name(perf_model_path)

        self.device = device
        self.bino = Bin(
            observer_name_or_path=obs_model_path,
            performer_name_or_path=perf_model_path,
        )

    def score(self, response_list):
        results = []
        for text in tqdm(response_list, desc="Binoculars evaluating"):
            results.append(self.bino.compute_score(text))
        return results


class SuperAnnotate(JudgeBase):
    # TODO: Fix Implimentation; Does not work atm
    def __init__(self, args):
        from utils.common import RobertaClassifier

        self.device = args.device
        self.model = RobertaClassifier.from_pretrained(
            "SuperAnnotate/ai-detector"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "SuperAnnotate/ai-detector"
        ).to(self.device)

    def score(self, response_list):
        results = []
        for text in tqdm(response_list, desc="SuperAnnotate evaluating"):
            tokens = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding="longest",
                truncation=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            _, logits = self.model(**tokens)
            proba = F.sigmoid(logits).squeeze(1).item()
            results.append(proba)
        return results

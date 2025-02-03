import os
import argparse
from utils.judges import load_judge
import json
import pandas as pd


def run(args):
    model = args.model
    prompt = args.prompt
    task = args.task
    output_file = args.output_file if args.output_file else args.data

    full_data = pd.read_csv(args.data)
    if model not in full_data["model"].unique() and model != "":
        raise ValueError("Model does not exist.")
    if prompt not in full_data["prompt"].unique() and prompt != "":
        raise ValueError("Prompt style does not exist.")
    if task not in full_data["task"].unique() and task != "":
        raise ValueError("Task does not exist.")

    data_to_score = full_data[
        ((full_data["model"] == model) | (model == ""))
        & ((full_data["prompt"] == prompt) | (prompt == ""))
        & ((full_data["task"] == task) | (task == ""))
    ].copy()

    if args.detector_args:
        detector_args = {
            key: value
            for key, value in [
                arg.split(":") for arg in args.detector_args.split(" ")
            ]
        }
    else:
        detector_args = {}

    judge = load_judge(args.judge_model, detector_args)

    responses = data_to_score["response"].tolist()
    judge_scores = judge.score(responses)

    data_to_score[args.judge_model] = judge_scores

    if output_file == args.data:
        # Fit the data back into the original dataframe
        full_data.loc[data_to_score.index, args.judge_model] = data_to_score[
            args.judge_model
        ]
        full_data.to_csv(output_file, index=False)
    elif args.append_output and os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        key_columns = ["task", "text_id", "prompt", "model"]

        existing_df_indexed = existing_df.set_index(key_columns)
        data_to_score_indexed = data_to_score.set_index(key_columns)

        all_columns = existing_df_indexed.columns.union(
            data_to_score_indexed.columns
        )
        existing_df_indexed = existing_df_indexed.reindex(columns=all_columns)
        data_to_score_indexed = data_to_score_indexed.reindex(
            columns=all_columns
        )

        existing_df_indexed.update(data_to_score_indexed)
        new_rows = data_to_score_indexed.loc[
            ~data_to_score_indexed.index.isin(existing_df_indexed.index)
        ]

        combined_df_indexed = pd.concat([existing_df_indexed, new_rows])
        combined_df = combined_df_indexed.reset_index()

        combined_df.to_csv(output_file, index=False)
    else:
        if args.append_output:
            print(f"Output file {output_file} does not exist. Creating...")
        data_to_score.to_csv(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Judge model parameters
    parser.add_argument(
        "--judge-model",
        help="Name of judge models",
        required=True,
    )

    parser.add_argument("--device", default="cuda", help="Device")

    parser.add_argument(
        "--detector-args",
        type=str,
        default="",
        help="Additional arguments for the detector wrapped"
        + ' in quotes "arg1:value1 arg2:value2"',
    )
    ##################################################

    # Data parameters
    parser.add_argument(
        "--data", required=True, help="Data to test on (in json)"
    )
    # will add Multilingual and Code gen
    parser.add_argument(
        "--task",
        default="",
        help="What task is being scored. Default includes all.",
    )

    parser.add_argument(
        "--model",
        default="",
        help="What generation model is being scored. Default includes all.",
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="What prompt style is being scored. Default includes all.",
    )
    ##################################################

    # Output parameters
    parser.add_argument(
        "--output-file",
        default="",
        help="Output file for the results.",
        type=str,
    )

    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to the output file instead of overwriting.",
    )

    ##################################################

    args = parser.parse_args()
    run(args)

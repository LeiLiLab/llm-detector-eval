import yaml
import subprocess
import sys
import torch

def run_detect(config, all_judge_configs):
    judges = config["judges"]
    if not type(judges) is list:
        raise ValueError("Must pass a list of judges.")
    task = config["task"]
    model = config["model"]
    prompt = config.get("prompt", None)
    data = config["data"]
    output_file = config.get("output_file", None)
    append_output = config.get("append_output", None)

    for judge_name in judges:
        judge_config = all_judge_configs.get(judge_name, None)
        cmd = [
            "python",
            "detect.py",
            "--data",
            data,
            "--task",
            task,
            "--model",
            model,
            "--judge-model",
            judge_name,
        ]
        if prompt:
            cmd.extend(["--prompt", prompt])
        if output_file:
            cmd.extend(["--output-file", output_file])
        if judge_config:
            cmd.extend(
                [
                    "--detector-args",
                    " ".join([f"{k}:{v}" for k, v in judge_config[0].items()]),
                ]
            )
        if append_output:
            cmd.extend(["--append-output"])
        print(f'Executing: {" ".join([str(item) for item in cmd])}')
        subprocess.run(cmd)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True


def run_generate(config):
    data = config["data"]
    task = config["task"]
    prompt = config["prompt"]
    model = config["model"]
    device = config.get("device", None)
    max_n_tokens = config.get("max_n_tokens", None)
    max_input_tokens = config.get("max_input_tokens", None)
    output_file = config.get("output_file", None)
    start = config.get("start", None)
    end = config.get("end", None)
    language = config.get("language", None)

    cmd = [
        "python",
        "generate.py",
        "--data",
        data,
        "--task",
        task,
        "--model",
        model,
        "--prompt",
        prompt,
    ]
    if device:
        cmd.extend(["--device", device])
    if max_n_tokens:
        cmd.extend(["--max-n-tokens", max_n_tokens])
    if max_input_tokens:
        cmd.extend(["--max-input-tokens", max_input_tokens])
    if output_file:
        cmd.extend(["--output-file", output_file])
    if start:
        cmd.extend(["--start", start])
    if end:
        cmd.extend(["--end", end])
    if start:
        cmd.extend(["--language", language])

    print(f'Executing: {" ".join([str(item) for item in cmd])}')
    subprocess.run(cmd)

    return True


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise ValueError("Must pass the config file and only the config file.")

    # Load configuration from the file
    with open(sys.argv[1], "r") as file:
        config = yaml.safe_load(file)

    # Iterate through experiments
    for experiment in config["experiments"]:
        exp_name = experiment["name"]
        exp_type = experiment["type"]
        print(f"Running Experiment: {exp_name}")
        if exp_type == "generate":
            run_generate(experiment)
        elif exp_type == "detect":
            run_detect(experiment, config["judges"])
        else:
            raise ValueError("Experiment Type doesn't exist.")


if __name__ == "__main__":
    main()

import argparse
import os
import traceback

from inferenceKit.model_config import supported_Reward_LLM, supported_SFT_LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_model", type=str, default=None)
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--cot_method", type=str, default="none")
    parser.add_argument("--step_method", type=str, default="vanilla")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = supported_SFT_LLM[args.model](model_path=args.model_path)
    cot_method = args.cot_method
    step_method = args.step_method
    generate_config = {"cot_method": cot_method, "step_method": step_method}
    reward_model = None
    if args.reward_model is not None:
        reward_model = (
            supported_Reward_LLM[args.reward_model](model_path=args.reward_model_path)
            if args.reward_model_path is not None
            else supported_Reward_LLM[args.reward_model]()
        )
    history = []
    while True:
        try:
            cur_input = input("Please enter input: ")
            if cur_input == "QUIT":
                break
            if cur_input == "CLEAR":
                history = []
            history.append(cur_input)
            response = model.generate(
                history, reward_model=reward_model, generate_config=generate_config
            )
            history.append(response)
            print("Assistant: {}\n".format(response))
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    main()

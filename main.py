import json
import subprocess
import sys
import os
from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.sanitize import script
from evalplus.evaluate import evaluate
import gen_solution 

def main():
    problems = get_human_eval_plus()

    if len(sys.argv) == 2:
        problem = problems[sys.argv[1]]
    else:
        print("This evaluation requires 2 arguments. Please provide a HumanEval problem as second argument. E.g. 'HumanEval/2'")
        return

    print(f"Solving Problem: {problem['prompt']}")

    #solution = gen_solution.gpt4o(problem["prompt"])
    solution = gen_solution.deepseek(problem["prompt"])

    # use BO to figure out a better or append existing prompt with new instruction: promt_aug
    # call LLM with this prompt_aug which will return code samples
    # 
    sample = {"task_id": sys.argv[1], "solution": solution}

    write_jsonl("results/samples.jsonl", [sample])
    print("Solutions saved to samples.jsonl")

    script("results/samples.jsonl")

    evaluate("humaneval", "results/samples-sanitized.jsonl")
    print("Code has been evaluated")

if __name__ == "__main__":
    main()

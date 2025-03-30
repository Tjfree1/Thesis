import openai
import json
import subprocess
import sys
import os
from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.sanitize import script
from evalplus.evaluate import evaluate

# Set your API key
client = openai.Client(api_key = os.environ.get("OPENAI_API_KEY"))

def gen_solution(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[{"role": "system", "content": "Please make a complete python code solution for the following problem. Do not include tests or excessive comments."}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating solution: {e}")
        return ""


def main():
    problems = get_human_eval_plus()

    if len(sys.argv) == 2:
        problem = problems[sys.argv[1]]
    else:
        print("This evaluation requires 2 arguments. Please provide a HumanEval problem as second argument. E.g. 'HumanEval/2'")
        return

    print(f"Solving Problem: {problem['prompt']}")

    solution = gen_solution(problem["prompt"])
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

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

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    file = open("results/metrics/passAtK.txt", "x")
    file.close()

    i=0

    for task_id, problem in problems.items():
        print(f"Solving Problem: {task_id}")

        try:
            solution = gen_solution.deepseek(problem["prompt"])
        except Exception as e:
            print(f"Error generating solution for {task_id}: {e}")
            continue

        sample = {"task_id": task_id, "solution": solution}
        task_id_sanitized = task_id.replace('/', '_')
        raw_file = f"results/{task_id_sanitized}.jsonl"
        sanitized_file = f"results/{task_id_sanitized}-sanitized.jsonl"

        # Write raw sample
        write_jsonl(raw_file, [sample])
        print(f"Saved to {raw_file}")

        try:
            script(raw_file)
            print(f"Sanitized {raw_file}")
        except Exception as e:
            print(f"Sanitization failed for {task_id}: {e}")
            continue

        try:
            sanitized_path = os.path.dirname(sanitized_file)
            eval_dir = f"results/HumanEval_{i}-sanitized.jsonl"
            evaluate("humaneval", eval_dir)

        except Exception as e:
            print(f"Evaluation failed for {task_id}: {e}")
        
        eval_result_path = f"results/HumanEval_{i}-sanitized.eval_results.json"
        try:
            with open(eval_result_path, "r") as f:
                data = json.load(f)
                base_pass_at_1 = data["pass_at_k"]["base"].get("pass@1", "N/A")
                plus_pass_at_1 = data["pass_at_k"]["plus"].get("pass@1", "N/A")
        except Exception as e:
            base_pass_at_1 = "ERROR"
            plus_pass_at_1 = "ERROR"
            print(f"Error reading eval results for {task_id}: {e}")

        with open("results/metrics/passAtK.txt", "a") as file:
            file.write(f"{task_id} | base pass@1: {base_pass_at_1}, plus pass@1: {plus_pass_at_1}\n")

        i += 1


if __name__ == "__main__":
    main()
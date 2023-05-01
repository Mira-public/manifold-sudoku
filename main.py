from solutions.catnee import catnee_prompt_1
from sudoku import collect_transition_rules_until_limit, execute_fixed_prompt, PROMPT_TOKENS, OUTPUT_TOKENS, TOTAL_TOKENS
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Solve Sudoku using GPT-4")

    parser.add_argument('--max-output-tokens', type=int, default=2048, help='Maximum number of output tokens')
    parser.add_argument('--max-turns', type=int, default=50, help='Maximum number of turns')
    parser.add_argument('--max-entries', type=int, default=200, help='Maximum number of entries to unroll the fixed prompt to')
    parser.add_argument('--output', type=str, default='sudoku_log.txt', help='Output log filename')
    parser.add_argument('--model', type=str, default='gpt-4', help='Model ID to use for GPT-4')
    parser.add_argument('--puzzle', type=str, required=True, help='Sudoku puzzle string')
    parser.add_argument('--prompt', type=str, required=True, help='Name of the fixed prompt')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of times to retry OpenAI requests')

    prompts = {
        'catnee1': catnee_prompt_1,
    }
    
    args = parser.parse_args()
    
    # Wipe the log file
    with open(args.output, 'w') as log_file:
        pass
    start_time = time.time()
    results = execute_fixed_prompt(prompts[args.prompt], args)
    print(f"Elapsed time={time.time() - start_time} PROMPT_TOKENS={results['prompt_tokens']} OUTPUT_TOKENS={results['output_tokens']} TOTAL_TOKENS={results['total_tokens']}")

if __name__ == "__main__":
    main()

# # # Collect transition rules
# transition_rules = collect_transition_rules_until_limit(catnee_prompt_1)

# # # Print the collected transition rules
# for rule in transition_rules:
#     print(rule)

# model = "gpt-3.5-turbo"
    
# entries = execute_fixed_prompt(
#     "075400001032000095000125080500230004208510000410008652300050920906040030000309508",
#     catnee_prompt_1,
#     model=model
#     )

# print(entries)

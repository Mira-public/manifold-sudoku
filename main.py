from solutions.catnee import catnee_prompt_1
from sudoku import collect_transition_rules_until_limit, execute_fixed_prompt, Checkpoint, find_solved_sudoku
import argparse
import time
import copy
import concurrent.futures
import os

puzzle_banks = {
    # https://www.latimes.com/games/sudoku
    "latimes-small": [
        "036004100051000070000019002027030601500000000003270058670150000100006280340000016", # May 1, 2023
        "090005020310906870860273010980037000000520180000400000024100390009060008000700400", # April 30, 2023
        "604000053000000092080400710800071500090305000025000070036014000100702000500806024", # April 29, 2023
        "400000237030005008800200495706300051040080003593706004000600100600840000908000000", # April 28, 2023
        "002001070907005080800070096003050240070000030201040000020530800089000000030009427", # April 27, 2023
        ]
}

def check_file(args):
    with open(args.file_path, "r") as file:
        file_content = file.read()

    potential_solution = find_solved_sudoku(file_content)

    if potential_solution:
        print(f"Found a potential solved Sudoku puzzle:\n{potential_solution}")
    else:
        print("No potential solved Sudoku puzzle found in the file.")

def main():
    parser = argparse.ArgumentParser(description="Solve Sudoku using GPT-4")
    subparsers = parser.add_subparsers()

    parser_run_prompt = subparsers.add_parser("run-prompt")
    parser_run_prompt.add_argument('--max-output-tokens', type=int, default=2048, help='Maximum number of output tokens')
    parser_run_prompt.add_argument('--max-turns', type=int, default=50, help='Maximum number of turns')
    parser_run_prompt.add_argument('--max-entries', type=int, default=200, help='Maximum number of entries to unroll the fixed prompt to')
    parser_run_prompt.add_argument('--output', type=str, default='sudoku_log.txt', help='Output log filename')
    parser_run_prompt.add_argument('--num-parallel', type=int, default=2, help='Maximum number of puzzles to be testing concurrently')
    parser_run_prompt.add_argument('--model', type=str, default='gpt-4', help='Model ID to use')
    parser_run_prompt.add_argument('--puzzle', type=str, required=True, help='Sudoku puzzle string')
    parser_run_prompt.add_argument('--prompt', type=str, required=True, help='Name of the fixed prompt')
    parser_run_prompt.add_argument('--max-retries', type=int, default=3, help='Maximum number of times to retry OpenAI requests')
    parser_run_prompt.add_argument('--stop-if-solved-puzzle-detected', type=bool, default=True, help='Use a hueristic to detected solved Sudokus and stop early')
    parser_run_prompt.set_defaults(func=run_prompt)
    
    # Create 'check-file' subcommand
    parser_check_file = subparsers.add_parser("check-file", help="Check if a file has something that looks like a solved Sudoku puzzle.")
    parser_check_file.add_argument("file_path", help="Path to the file to check.")
    parser_check_file.set_defaults(func=check_file)

    args = parser.parse_args()
    func = args.func
    del args.func
    func(args)
    
def run_prompt(args):
    prompts = {
        'catnee-1': catnee_prompt_1,
    }

    puzzles =  puzzle_banks[args.puzzle] if args.puzzle in puzzle_banks else args.puzzle.split(",")
    fixed_prompt = prompts[args.prompt]
    start_time = time.time()
    evaluate_multiple_puzzles(puzzles, fixed_prompt, args)

    print(f"Done with all puzzles. Elapsed time={time.time() - start_time}\n")

def evaluate_multiple_puzzles(puzzles, fixed_prompt, args, max_workers=5):
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_puzzle, puzzle, fixed_prompt, args) for puzzle in puzzles]

        for future in concurrent.futures.as_completed(futures):
            #try:
                result = future.result()
                results.append(result)
            #except Exception as e:
            #    print(f"An error occurred during evaluation: {e}")

    return results
    
def evaluate_puzzle(puzzle, fixed_prompt, args):
    args = copy.deepcopy(args)
    args.puzzle = puzzle
    args.output += "." + puzzle
    args.checkpoint = puzzle + '.ckpt'
    if os.path.exists(args.checkpoint):
        with open(args.output, 'a') as log_file:
            log_file.write("Resuming from checkpoint")
        checkpoint = Checkpoint.load(args.checkpoint)
        checkpoint.args = args # New command line arguments take priority
    else:
        # Wipe the log file
        with open(args.output, 'w') as log_file:
            log_file.write(f"Creating fresh checkpoint for puzzle {args.puzzle}\n")
        checkpoint = Checkpoint()
        checkpoint.args = args

    start_time = time.time()
    result = execute_fixed_prompt(checkpoint, fixed_prompt, args)
    statistics = result["statistics"]
    with open(args.output, 'a') as log_file:
        log_file.write(f"Done. Elapsed time:{time.time() - start_time} Estimated cost:{statistics.total_cost(args.model)} Prompt tokens:{statistics.prompt_tokens} Output tokes:{statistics.output_tokens} Total tokens:{statistics.total_tokens}\n")
    
    
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

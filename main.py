from solutions.catnee import catnee_prompt_1
from solutions.joshua import joshua_prompt_1
from solutions.peter import peter_prompt_1
from manifold_sudoku import collect_transition_rules_until_limit, execute_fixed_prompt, Checkpoint, find_solved_sudoku, string_to_visual_representation, solve_puzzle
import argparse
import time
import copy
import concurrent.futures
import os
from pathlib import Path

puzzle_banks = {
    # https://www.latimes.com/games/sudoku
    "latimes-small": [
        "036004100051000070000019002027030601500000000003270058670150000100006280340000016", # May 1, 2023
        "090005020310906870860273010980037000000520180000400000024100390009060008000700400", # April 30, 2023
        "604000053000000092080400710800071500090305000025000070036014000100702000500806024", # April 29, 2023
        "400000237030005008800200495706300051040080003593706004000600100600840000908000000", # April 28, 2023
        "002001070907005080800070096003050240070000030201040000020530800089000000030009427", # April 27, 2023
        ],
    "latimes-medium": [
        "051000307009700000437029056000503900300000080748061205912008000000002068063017009", # June 23, 2023
        "706000495031209000490000002042506100000008009010932604370621008000350270060007031", # June 24, 2023
        "450290786060000003100070409930024500006000132000800070070600805015400097208005000", # June 25, 2023
        "431000000600040000000090100047200013006730009000450062020900600304065201168027054", # June 26, 2023
        "600714030438200601090600000069305047543000062070000500000000309002008000906000720", # June 27, 2023
        "900562074060000000504100008000059060030480720005000019206070000390810200408003907", # June 28, 2023
        "000203004070000000000401597803560420010000678060000050140020830600050042285700910", # June 29, 2023
        "030004500000070020401900000087400050050100004004000870005200060690000048872009000", # June 30, 2023
        "029000803005007000080053172104000006000100285060000400200800000537601920090030504", # July 1, 2023
        "000168000760000000003900456009015000054700201307024500500400670900007100270001390", # July 2, 2023
        "090300200045680973008000061900006708002008050070090124700400000009701800004060090", # July 3, 2023
        "500930000890062400020000000000006500005400900000020031001840205002010840038050600", # July 4, 2023
        "000800020430260000000004030970038042001572960325000800000007306000401050500000170", # July 5, 2023
        "005090000700600589400105230840020060000010708000003900016008400008040090390067805", # July 6, 2023
        "003960082491008006000000010009647500000009160004100790320090600070020049908010000", # July 7, 2023
        "603007001590834002200000049018070035005009028060580100020060093000300007309450216", # July 8, 2023
        "080670030000002800670930140560090710910400000000001060004186002000009000300720001", # July 9, 2023
        "400890050026405008500010402800040000139080260200009001050030049094100000300000506", # July 10, 2023
        "004072108000038000830645720000510072090200360050003401900057000203400850400000000", # July 11, 2023
        "010060000000300007000791086035200048008400960104006500420000000081650203006103870", # July 12, 2023
    ],
    "latimes-october": [
        "041670258000000003700052600204000080000000064010030020030080490092041000060709005", # October 1, 2023
        "000240000092005300000000081583090720000780500740002000000916003301408065008507000", # October 2, 2023
        "000394650060000003008150000039007000457002060800900014000000080900061000015280046", # October 3, 2023
        "097014002030000900026500080000000030602308100010050047009000001365002490000089020", # October 4, 2023
        "050004000000900370900801000803006100060010827091008600700080903000290004030160500", # October 5, 2023
        "458200090960070030002010605000350971000001860210009003023004000146583020090020004", # October 6, 2023
        "081023006300590807060080002000050001013000700200600900509002030104800560030910004", # October 7, 2023
        "160903400300070000970080350097040028400000000003100674000806502035204790216000000", # October 8, 2023
        "031790008005014090000620105000140000100509204000208006008461000060000901010080047", # October 9, 2023
        ],
}

def check_file(args):
    with open(args.file_path, "r") as file:
        file_content = file.read()

    potential_solution = find_solved_sudoku(file_content)

    if potential_solution:
        print(f"Found a potential solved Sudoku puzzle:\n{potential_solution}")
    else:
        print("No potential solved Sudoku puzzle found in the file.")

def check_puzzle(args):
    bad_puzzle = False
    for k,puzzles in puzzle_banks.items():
        for puzzle in puzzles:
            if solve_puzzle(puzzle):
                continue
            else:
                bad_puzzle = True
                print("Bad puzzle:", puzzle)
    if not bad_puzzle:
        print("All puzzles good.")

def print_puzzle(args):
    print(string_to_visual_representation(args.puzzle))
        
def main():
    parser = argparse.ArgumentParser(description="Solve Sudoku using GPT-4")
    subparsers = parser.add_subparsers()

    parser_run_prompt = subparsers.add_parser("run-prompt")
    parser_run_prompt.add_argument('--log-dir', type=str, default="outputs", help="Directory to store outputs")
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

    # Create 'print-puzzle' subcommand
    parser_print_puzzle = subparsers.add_parser("print-puzzle", help="Print the puzzle")
    parser_print_puzzle.add_argument('--puzzle', type=str, required=True)
    parser_print_puzzle.set_defaults(func=print_puzzle)

    # Create 'check-puzzle' subcommand
    parser_check_puzzle = subparsers.add_parser("check-puzzle", help="Check the puzzle bank")
    parser_check_puzzle.add_argument('--puzzle', type=str)
    parser_check_puzzle.set_defaults(func=check_puzzle)
    
    args = parser.parse_args()
    func = args.func
    del args.func
    func(args)
    
def run_prompt(args):
    prompts = {
        'catnee-1': catnee_prompt_1,
        'joshua-1': joshua_prompt_1,
        'peter-1': peter_prompt_1,
    }

    puzzles =  puzzle_banks[args.puzzle] if args.puzzle in puzzle_banks else args.puzzle.split(",")
    fixed_prompt = prompts[args.prompt]
    start_time = time.time()
    evaluate_multiple_puzzles(puzzles, fixed_prompt, args)

    print(f"Done with all puzzles. Elapsed time={time.time() - start_time}\n")

def evaluate_multiple_puzzles(puzzles, fixed_prompt, args):
    results = []

    max_workers = args.num_parallel
    
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
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(log_dir / (args.prompt + "." + puzzle + "." + args.output))
    args.checkpoint = str(log_dir / (args.prompt + "." + puzzle + '.ckpt'))
    if Path(args.checkpoint).exists():
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

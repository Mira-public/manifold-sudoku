from solutions.catnee import catnee_prompt_1
from solutions.joshua import joshua_prompt_1
from solutions.peter import peter_prompt_1
from solutions.emily.emily import emily_prompt_2, emily_simulate_run

from manifold_sudoku import collect_transition_rules_until_limit, execute_fixed_prompt, Checkpoint, find_solved_sudoku, string_to_visual_representation, solve_puzzle, PuzzleSolution, load_cache, UnsolvablePuzzle, grid_to_string, rotate_sudoku, rotate_sudoku_emily, extract_sudoku, find_problem_in_sudoku, MODEL_INFOS, get_transition_rules
import argparse
import time
import copy
import concurrent.futures
import os
from pathlib import Path
import re
from termcolor import colored
import sys

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
        "028017904600480000000020610010050070070061490009000003700008240053000000184000065", # October 10, 2023
        "002000347030140506500003008000800000200005009090204065005000801900000700871050694", # October 11, 2023
        "000000300900100056000735090059062040321400500000003000690007001147200030000000089", # October 12, 2023
        "076040008000100900401908076100560000307402069009080100800601400005030680614000390", # October 13, 2023
        "006028000052000610040000000000903580008406090304005000000000073260031840703500020", # October 14, 2023
        "036000902008500410000080600803000107900017304710030000307651040095740200000900706", # October 15, 2023
        "000400010600502000450080690000000082026970000183200900001053400308620050205810706", # October 16, 2023
        "007630210014590000200070900000000170300710009000000603020000005600208090700340800", # October 17, 2023
        "431080600702004300000020010005003900093508007100006500000000006920000005057000100", # October 18, 2023
        "702000300640038057050710000076020180814906030030080506000041260108005000000370000", # October 19, 2023
        "018090050030100640006035008704513900000270010052000034205060001691000000003951006", # October 20, 2023
        "000000083000000000108300470030810900020900068059004007000080006370060024080240030", # October 21, 2023
        "006700240000080006080200007000600081204978650800001902500000000100869000028450709", # October 22, 2023
        "000300079006009100100050006700000030800000905390208000610905040250043001007001020", # October 23, 2023
        "000703068760208054089050000200900640050012007890400205000009000048571026072000080", # October 24, 2023
        "650070004489000637300090500800006720971000300063507000000001473000305906030749008", # October 25, 2023
        "060050000284007300000014208509000700000006103002070000920705000001200580850031000", # October 26, 2023
        "408060500005200006016000302284090000703005100150700064802901050090030040007600020", # October 27, 2023
        "007059108104720000080016005000045020210800050405200906390060700560030800001000560", # October 28, 2023
        "016000530020006098903008000007000000200645087000079004580001300700463800061507020", # October 29, 2023
        "974800005620400030000259000480000090300607040790000003008910070047005600030086014", # October 30, 2023
        "530004100007000023019005084001020078000003950000000231950067800280100600070042000", # October 31, 2023
        ],
    "latimes-december": [
        "208400705563000000407300600600000081040610007080040300009000003006001270034250108", # December 1, 2023
        "450000060189000400200005800001429708008060005004080200000370020902006001007010084", # December 2, 2023
        "700000300014090070069002154000000047400327900050061038005200090243050000000708003", # December 3, 2023
        "050300010107208340000000007000893000970005003003140960001400700002506401405009000", # December 4, 2023
        "000600098002700006670900240407000000009040507000508400928000000060239070005000900", # December 5, 2023
        "700302819190086037003000000600425080000073604030000005000800000015900300007000540", # December 6, 2023
        "034720009608000250700900003910000765020300800070009300080001504001003000509402030", # December 7, 2023
        "703001000014823000250700004031008507000002010009000000002070080000304700307009056", # December 8, 2023
        "620953170500004300090001640480000500200000417071005980000030204700000891000106000", # December 9, 2023
        "600400020002008560704002000800300056910084003500006900360000090090607030000053601", # December 10, 2023
        ],
    "latimes-january": [
        "203090450400100970901247300005063080096084000704510009010400593630800040002009006", # January 1, 2023
        "200043700700000005501000409016304000405901600900008100090030500004000006687420090", # January 2, 2023
        "200760050900100000006208704090405800520870010007010420060000500400087001730092040", # January 3, 2023
        "000730000307800504190502000013000720700005900580009043005680179801007000409000806", # January 4, 2023
        "040903860803040091000705000390200000457001020100000040506307910000008673730190080", # January 5, 2023
        "049070020015000980000950670050200700070000003200500416031068500060400030500009060", # January 6, 2023
        "907000200130502040450080017500000090009003006070000035000027064005061020700305000", # January 7, 2023
        "908134002105200070004000000000600009400070205216000000009008024050000300700395800", # January 8, 2023
        "040000600000009378076000002009007104004510200200090050830002000497008000062701093", # January 9, 2023
        "792600004000107503000009602005000081027410050308090400000064020070000105200070046", # January 10, 2023
        "010009700700000804400023600030070401070600385005100097167350000083900026000016000", # January 11, 2023
        "509002614140300725620400309800700200051000470200050030410007000300000500005086140", # January 12, 2023
        "100007005209180730070095600408070260706300400300860070000700010037000080000020900", # January 13, 2023
        "042038090608400003930060082200050917005020004390000020000000170000000060009615300", # January 14, 2023
    ],
    "latimes-january-medium": [
        "003800000020000300000600002005000040130049000702000100050091000000070090900508006", # January 1, 2023
        ],
    "latimes-january-hard": [
        "003042670004000200590000000000006300006028010000500002400000000020065704050130000", # January 1, 2023
        ],
    "latimes-january-expert": [
        "000500040300492000100000500010030005908000102000000006060009000501080960200307000", # January 1, 2023
        ],
    "nytimes": [
        "075396000000050209968000057430600810600543000009100603007005026096002030500061070", # October 17, 2023
        ],
}

SOLUTION_REGEXES = {
    'default': r"([1-9][0\-_|\+\s\n]*?){81}",
    'emily-2': r"<output>\s*(?:Row\w+: \[(\d),(\d),(\d),(\d),(\d),(\d),(\d),(\d),(\d)\]\s*){9}</output>",
    }

SOLUTION_CHECKS = {
    'emily-2': emily_simulate_run,
    }

PROMPTS = {
    'catnee-1': catnee_prompt_1,
    'joshua-1': joshua_prompt_1,
    'peter-1': peter_prompt_1,
    #'emily-1': emily_prompt_1,
    'emily-2': emily_prompt_2,
    #"oztelle": oztelle_prompt,
}

def check_file(args):
    with open(args.file_path, "r") as file:
        file_content = file.read()

    potential_solution = find_solved_sudoku(file_content)

    if potential_solution:
        print(f"Found a potential solved Sudoku puzzle:\n{potential_solution}")
    else:
        print("No potential solved Sudoku puzzle found in the file.")

def read_stdin_as_string():
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip())

    return '\n'.join(lines)
        
def check_puzzle(args):
    bad_puzzle = False
    for k,puzzles in puzzle_banks.items():
        for puzzle in puzzles:
            if solve_puzzle(puzzle):
                continue
            else:
                bad_puzzle = True
                print("Bad puzzle:", puzzle)
    if args.puzzle:
        for k in SOLUTION_CHECKS:
            check_function = SOLUTION_CHECKS[k]
            _,_,_,solved = check_function(args.puzzle, print_enabled=False)
            if solved < 0:
                print(f"Puzzle unsolvable with technique {k}: {args.puzzle}")
        if args.puzzle == "input":
            args.puzzle = read_stdin_as_string()
            args.puzzle = extract_sudoku(args.puzzle)
        if not solve_puzzle(args.puzzle):
            bad_puzzle = True
            print("Bad argument puzzle", find_problem_in_sudoku(args.puzzle))
    if not bad_puzzle:
        print("All puzzles good.")

def print_puzzle(args):
    print(string_to_visual_representation(args.puzzle))

def print_transitions(args):
    prompt = PROMPTS[args.prompt]
    transition_rules = get_transition_rules(0, emily_prompt_2, args)
    for i, x in enumerate(transition_rules):
        print(f"{i}: {x}")
    
def main():
    load_cache()
    parser = argparse.ArgumentParser(description="Solve Sudoku using GPT-4")
    subparsers = parser.add_subparsers()

    # Create 'print-transitions' subcommmand
    parser_print_transitions = subparsers.add_parser('print-transitions', help="Print transition rules in plaintext")
    parser_print_transitions.add_argument("prompt", help="Prompt to print transition rules for")
    parser_print_transitions.add_argument("--max_turns", default=50)
    parser_print_transitions.add_argument("--max_transitions", default=1000)
    parser_print_transitions.set_defaults(func=print_transitions)

    # 'run-prompt' command
    parser_run_prompt = subparsers.add_parser("run-prompt")
    parser_run_prompt.add_argument('--log-style', type=str, default="mira", help="Log style")
    parser_run_prompt.add_argument('--log-dir', type=str, default="outputs", help="Directory to store outputs")
    parser_run_prompt.add_argument('--max-output-tokens-per-request', type=int, default=None, help='Maximum number of output turns per API request')
    parser_run_prompt.add_argument('--max-output-tokens', type=int, default=None, help='Maximum number of output tokens total. Multiple requests will be made summing to this.')
    parser_run_prompt.add_argument('--max-turns', type=int, default=50, help='Maximum number of turns')
    parser_run_prompt.add_argument('--max-transitions', type=int, default=200, help='Maximum number of transition rules to unroll the fixed prompt to')
    parser_run_prompt.add_argument('--output', type=str, default='sudoku_log.txt', help='Output log filename')
    parser_run_prompt.add_argument('--num-parallel', type=int, default=2, help='Maximum number of puzzles to be testing concurrently')
    parser_run_prompt.add_argument('--model', type=str, default='gpt-4-1106-preview', help='Model ID to use')
    parser_run_prompt.add_argument('--puzzle', type=str, required=True, help='Sudoku puzzle string')
    parser_run_prompt.add_argument('--prompt', type=str, required=True, help='Name of the fixed prompt')
    parser_run_prompt.add_argument('--max-retries', type=int, default=3, help='Maximum number of times to retry OpenAI requests or continue a chunked request')
    parser_run_prompt.add_argument('--require-solvable-puzzle', type=int, default=2, help='Every nth model response should have a solvable puzzle in its output.')
    parser_run_prompt.add_argument('--stop-if-solved-puzzle-detected', type=bool, default=True, help='Use a hueristic to detected solved Sudokus and stop early')
    parser_run_prompt.add_argument('--skip-invalid-puzzle-check', action="store_true", help='Check that puzzles match a pattern')
    parser_run_prompt.add_argument('--use-checkpoint', type=bool, default=False, help="Whether to load a checkpoint file or start from scratch")
    parser_run_prompt.add_argument('--seed', type=int, default=1)
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
    puzzles =  puzzle_banks[args.puzzle] if args.puzzle in puzzle_banks else args.puzzle.split(",")
    if args.prompt in SOLUTION_CHECKS:
        check_function = SOLUTION_CHECKS[args.prompt]
        for puzzle in puzzles:
            _,_,_,solved = check_function(args.puzzle, print_enabled=False)
            if solved < 0:
                print(f"Puzzle unsolvable with technique {args.prompt}: {args.puzzle}")
                return
    fixed_prompt = PROMPTS[args.prompt]
    args.solution_pattern = SOLUTION_REGEXES[args.prompt] if args.prompt in SOLUTION_REGEXES else SOLUTION_REGEXES['default']
    model_info = MODEL_INFOS[args.model]
    if args.max_output_tokens_per_request is None:
        args.max_output_tokens_per_request = model_info["output_tokens"]
    if args.max_output_tokens is None:
        args.max_output_tokens = model_info["context_window"]
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

def diff_strings(str1, str2, color='red'):
    if len(str1) != len(str2):
        return "Error: Strings must be of equal length"
    
    highlighted_diff = ""
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            highlighted_diff += colored(char2, color)
        else:
            highlighted_diff += char2
    return highlighted_diff

def diff_strings(str1, str2, color='red', ignore_chars='0'):
    if len(str1) != len(str2):
        return "Error: Strings must be of equal length"
    
    highlighted_diff = ""
    for char1, char2 in zip(str1, str2):
        skip = char1 in ignore_chars or char2 in ignore_chars
        if char1 != char2 and not skip:
            highlighted_diff += colored(char2, color)
        else:
            highlighted_diff += char2
    return highlighted_diff

def evaluate_puzzle(puzzle, fixed_prompt, args):
    args = copy.deepcopy(args)
    args.puzzle = puzzle
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(log_dir / (args.prompt + "." + puzzle + "." + args.output))
    args.checkpoint = str(log_dir / (args.prompt + "." + puzzle + '.ckpt'))
    if args.use_checkpoint and Path(args.checkpoint).exists():
        with open(args.output, 'a') as log_file:
            log_file.write("Resuming from checkpoint")
        checkpoint = Checkpoint.load(args.checkpoint)
        checkpoint.args = args # New command line arguments take priority
        Checkpoint.print_checkpoint(checkpoint.conversation)
    else:
        # Wipe the log file
        with open(args.output, 'w') as log_file:
            if args.log_style == "mira":
                log_file.write(f"Creating fresh checkpoint for puzzle {args.puzzle}\n")
        checkpoint = Checkpoint()
        checkpoint.args = args
    def print_history(checkpoint, candidate=None, file=sys.stderr):
        solved_board = solve_puzzle(checkpoint.args.puzzle)
        solved = grid_to_string(solved_board)
        if candidate:
            candidate = rotate_sudoku_emily(candidate, len(checkpoint.solution_history))
            print(f"S: {solved}", file=file)
            print(f"C: {diff_strings(solved, candidate)}", file=file)
        for i, p in enumerate(checkpoint.solution_history, 1):
            p = rotate_sudoku_emily(p, i) # Emily's puzzles are rotated
            #print(diff_strings(p, e.checkpoint.args.puzzle))
            #print(diff_strings(e.checkpoint.args.puzzle, p))
            print(f"{i}: {diff_strings(checkpoint.args.puzzle, p)}", file=file)
        
    start_time = time.time()
    try:
        result = execute_fixed_prompt(checkpoint, fixed_prompt, args)
        result_checkpoint = result["statistics"]
        print_history(checkpoint)
    except PuzzleSolution as e:
        result_checkpoint = e.checkpoint
        print(f"Early-stopping with solution: {e.solution}")
    except UnsolvablePuzzle as e:
        result_checkpoint = e.checkpoint
        print(f"Early-stopping because the following solution checkpoint is unsolvable:")
        print_history(e.checkpoint, candidate=e.unsolvable)


    statistics = result_checkpoint
    with open(args.output, 'a') as log_file:
        print_history(result_checkpoint, file=log_file)
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

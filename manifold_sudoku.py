#import openai
import json
import time
import os
import requests
import datetime
from dataclasses import dataclass
import re
from sudoku import Sudoku
import tiktoken
from typing import Callable, Any
import inspect
import itertools
import openai
import hashlib

def convert_pairs_to_openai(entries):
    formatted_messages = [{"role": role, "content": content} for role, content in entries]
    return formatted_messages

def openai_api_key():
    return os.environ.get("OPENAI_API_KEY")

def extract_sudoku(text):
    cell_pattern = r"\D*(\d)"
    sudoku_pattern = cell_pattern*81 + r"\D*"
    mat = re.search(sudoku_pattern, text)
    return ''.join(mat.groups())

def find_solved_sudoku(pattern, text):
    mat = re.search(pattern, text)
    if mat:
        prematch = mat.group()
        print(f"@@@@ PREMATCH={prematch}")
        return extract_sudoku(prematch)
    else:
        return None

@dataclass
class Checkpoint:
    args = None
    transition_index: int = 0
    turn_number: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    conversation = []
    solution_history = []
    
    def total_cost(self, model):
        if model == "gpt-3.5-turbo":
            return 0.002 * self.total_tokens / 1000. # $0.002 / 1k tokens
        elif model == "gpt-4":
            return 0.03 * self.prompt_tokens / 1000. + 0.06 * self.output_tokens / 1000.

    def serializable(self):
        checkpoint = {
            "args": vars(self.args),
            "transition_index": self.transition_index,
            "turn_number": self.turn_number,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "conversation": self.conversation,
            "solution_history": self.solution_history,
        }
        return checkpoint
        
    def save(self):
        checkpoint = self.serializable()
        with open(self.args.checkpoint, 'w') as f:
            return json.dump(checkpoint, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            print(f"About to load {filename}")
            checkpoint = json.load(f)
        ckpt = cls()
        ckpt.args = checkpoint["args"]
        ckpt.transition_index = checkpoint["transition_index"]
        ckpt.turn_number = checkpoint["turn_number"]
        ckpt.prompt_tokens = checkpoint["prompt_tokens"]
        ckpt.output_tokens = checkpoint["output_tokens"]
        ckpt.total_tokens = checkpoint["total_tokens"]
        ckpt.conversation = checkpoint["conversation"]
        ckpt.solution_history = checkpoint["solution_history"]
        return ckpt

    @classmethod
    def print_checkpoint(cls, entries):
        print("Conversation token counts")
        for i, entry in enumerate(entries):
            print(entry)
            token_count = gpt4_enc.encode(entry[1])
            print(f"{i}: {len(token_count)} tokens")
    
def solve_puzzle(puzzle_string):
    board = string_to_list_of_lists(puzzle_string)
    sudoku = Sudoku(3,3,board=board)
    try:
        solved_board = sudoku.solve(raising=True).board
        #print(solved_board)
        return solved_board
    except:
        return None

def rotate_sudoku(puzzle, di):
    di = di%81
    assert di >= 0
    return puzzle[-di:] + puzzle[:-di]

def rotate_sudoku_emily(puzzle, turn_number):
    return rotate_sudoku(puzzle, 27*(turn_number+1))

# The official Python bindings were taking like 3 minutes for some reason, so just POST the API directly.
def openai_chat_completion(messages, args, n=1):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key()}",
    }
    payload = {
        "model": args.model,
        "messages": messages,
        "n": n,
        "max_tokens": args.max_output_tokens,
        "temperature": 0,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    for attempt in range(args.max_retries):
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            return response.json()
        else:
            if attempt < args.max_retries - 1:
                print("Request failed. Sleeping and then retrying")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise Exception(f"OpenAI API request failed after {args.max_retries} attempts with status code {response.status_code}: {response.text}")

CACHE_FILE = "cache.json"
CACHE = {}

def get_hash(x):
    return hashlib.sha256(json.dumps(x).encode('utf-8')).hexdigest()

def load_cache():
    global CACHE
    try:
        with open(CACHE_FILE, 'r') as f:
            CACHE = json.load(f)
        print(f"Loaded response cache with {len(CACHE)} entries")
    except FileNotFoundError:
        print("No existing cache detected.\n")
    except e:
        print(f"Received error loading cache: {e}")

def save_cache():
    global CACHE
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(CACHE, f, sort_keys=True)

def get_cache(key):
    k = get_hash(key)
    return CACHE.get(k)

def set_cache(key, value):
    global CACHE
    k = get_hash(key)
    CACHE[k] = value
    save_cache()

# Remove args that shouldn't change the result of any single invocation to GPT-4
def important_args(args):
    vs = vars(args)
    return {
        "max-output-tokens": vs["max_output_tokens"],
        "puzzle": vs["puzzle"],
        "prompt": vs["prompt"],
        "model": vs["model"],
        }
    
def count_conversation_tokens(entries):
    token_counts = [len(gpt4_enc.encode(message)) for (_,message) in entries]
    message_tokens = sum(token_counts)
    speaker_tokens = len(entries)
    padding_tokens = 15 + 12 # TODO: Figure out where the 15 extra tokens are coming from in Emily's prompt A and 12 in prompt B
    return {
        "token_counts": token_counts,
        "message_tokens": message_tokens,
        "speaker_tokens": speaker_tokens,
        "padding_tokens": padding_tokens,
        "total_tokens": message_tokens + speaker_tokens + padding_tokens,
        }
    
def run_gpt_4(entries, args, statistics):
    if args.model == 'mock':
        return "mock GPT-4 string" # Use to test without hitting API
    cache_key = {"conversation": entries, "args": important_args(args)}
    c = get_cache(cache_key)
    if c is not None:
        response = c
    else:
        token_stats = count_conversation_tokens(entries)
        max_tokens = args.max_output_tokens
        max_output_tokens = max_tokens - token_stats["total_tokens"]
        print(f"About to run {args.model} with {max_tokens}={max_output_tokens}+sum({token_stats['token_counts']})+{token_stats['speaker_tokens']}+{token_stats['padding_tokens']}")
        print(f"{len(entries)} Entries: {entries}")
        start_time = time.time()
        for i in range(args.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=args.model,
                    max_tokens=max_output_tokens,
                    n=1,
                    temperature=0,
                    messages=convert_pairs_to_openai(entries)
                )
                break
            except openai.error.Timeout as e:
                print(f"Received timeout: {e}")
                response = None
                continue
            except openai.error.InvalidRequestError as e:
                Checkpoint.print_checkpoint(entries)
                raise e
        if response is None:
            raise Exception("Unable to get a response after {args.max_retries} attempts")
            
        #openai_entries = convert_pairs_to_openai(entries)
        # response = openai_chat_completion(openai_entries, args)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for {args.model} call: {elapsed_time:.2f} seconds.\n")
        if max_output_tokens == response["usage"]["completion_tokens"]:
            print("@@@@ RESPONSE:", response)
            raise Exception(f"Received exactly {max_output_tokens} tokens. Exiting because you may have hit the limit accidentally.")
        set_cache(cache_key, response)
    statistics.total_tokens += response["usage"]["total_tokens"]
    statistics.output_tokens += response["usage"]["completion_tokens"]
    statistics.prompt_tokens += response["usage"]["prompt_tokens"]
    return response["choices"][0]["message"]["content"].strip()

def collect_transition_rules_until_limit(fixed_prompt_function, response_limit=50, total_limit=200):
    transition_rules = []
    response_count = 0
    index = 0

    while response_count < response_limit and index < total_limit:
        rule = fixed_prompt_function(index)
        if rule[0] == "InsertResponse":
            response_count += 1
        transition_rules.append(rule)
        index += 1

    return transition_rules

def grid_to_string(board):
    return ''.join([str(c) for row in board for c in row])


def string_to_list_of_lists(puzzle_string):
    return [[int(puzzle_string[i * 9 + j]) for j in range(9)] for i in range(9)]

def string_to_list_of_strings(puzzle_string):
    return [puzzle_string[i * 9:(i + 1) * 9] for i in range(9)]

def string_to_multiline_string(puzzle_string):
    return "\n".join(string_to_list_of_strings(puzzle_string))

def string_to_visual_representation(puzzle_string):
    rows = [puzzle_string[i * 9:(i + 1) * 9] for i in range(9)]
    
    visual_representation = ""
    for i, row in enumerate(rows):
        if i % 3 == 0 and i > 0:
            visual_representation += "-+------+------+------\n"
        visual_row = ""
        for j, cell in enumerate(row):
            if j % 3 == 0 and j > 0:
                visual_row += "| "
            visual_row += cell if cell != '0' else '_'
            visual_row += ' '
        visual_representation += visual_row.rstrip() + '\n'
    return visual_representation

def string_to_2d_representation_no_bars(puzzle):
    xss = string_to_list_of_lists(puzzle)
    representation = ""
    for xs in xss:
        representation += " ".join([str(x) for x in xs])
        representation += "\n"
    return representation

enc = tiktoken.get_encoding("cl100k_base")

@dataclass
class Insert:
    index: int
    message: str
    tag: str = "user"

@dataclass
class Remove:
    index: int

@dataclass
class InsertPuzzle:
    index: int
    render_func: Callable[[str], str]

@dataclass
class InsertResponse:
    index: int

gpt4_enc = tiktoken.get_encoding("cl100k_base")
    
@dataclass
class Truncate:
    index: int
    num_tokens: int
    tokenizer: Any = gpt4_enc
    
def truncate_to_last_n_tokens(text, n, encoding):
    """
    Truncate text to the last n tokens.
    
    :param text: str, the input text
    :param n: int, the number of tokens to truncate to
    :param encoding: tiktoken encoding, the tokenizer model
    :return: str, the truncated text
    """
    # Tokenize the input text
    tokens = list(encoding.encode(text))
    
    # Truncate to the last n tokens
    truncated_tokens = tokens[-n:]
    
    # Decode back to text
    truncated_text = encoding.decode(truncated_tokens)
    
    return truncated_text

@dataclass
class PuzzleSolution(Exception):
    checkpoint : Any
    solution : str

@dataclass
class UnsolvablePuzzle(Exception):
    checkpoint : Any
    unsolvable : str

def apply_transition_rule(checkpoint, transition_rule, args):
    def conv_token_counts(pair):
        return len(gpt4_enc.encode(pair[1]))
    print(f"Turn {checkpoint.turn_number} rule {checkpoint.transition_index} on {len(checkpoint.conversation)} entries with tokens {list(map(conv_token_counts, checkpoint.conversation))}: {transition_rule}")
    checkpoint.transition_index += 1
    def translate_index(index):
        if index < 0:
            return index + len(checkpoint.conversation)+1
        else:
            return index
    def insert(index, role, message):
        index = translate_index(index)
        checkpoint.conversation.insert(index, (role, message))
    def remove(index):
        checkpoint.conversation.pop(index)
    def checkpoint_solution(puzzle):
        print(f"SOLUTION CHECKPOINT:{puzzle}")
        checkpoint.solution_history.append(puzzle)
    match transition_rule:
        case Remove(index):
            index = translate_index(index)
            checkpoint.conversation.pop(index)
        case Insert(index, message, tag):
            index = translate_index(index)
            insert(index, tag, message)
        case InsertPuzzle(index, render_func):
            index = translate_index(index)
            rendered_puzzle = render_func(args.puzzle)
            insert(index, "user", rendered_puzzle)
        case InsertResponse(index):
            index = translate_index(index)
            response = run_gpt_4(checkpoint.conversation, args, checkpoint)
            insert(index, "assistant", response)
            checkpoint.turn_number += 1
            checkpoint.save() # Long-running API call
            match args.log_style:
                case "mira":
                    log_conversation(checkpoint, args.output)
                case "emily":
                    log_conversation_emily(checkpoint, args.output)
            potential_solution = find_solved_sudoku(args.solution_pattern, response)
            if not potential_solution and checkpoint.turn_number % args.require_solvable_puzzle == 0:
                raise Exception(f"No puzzle pound in {response}")
            if potential_solution:
                checkpoint_solution(potential_solution)
                is_complete = "0" not in potential_solution
                if is_complete:
                    print(f"POTENTIAL SOLUTION:{potential_solution}")
                    if args.stop_if_solved_puzzle_detected:
                        solution = solve_puzzle(potential_solution)
                        if solution:
                            print("Early-stopping with valid solution")
                            raise PuzzleSolution(checkpoint, solution)
                        else:
                            raise Exception(f"Unsolvable puzzle: {potential_solution}")
                else:
                    solution = solve_puzzle(potential_solution)
                    #print("@@@@@@@", potential_solution, solution)
                    if not solution:
                        raise UnsolvablePuzzle(checkpoint, potential_solution)
                        
        case Truncate(index, num_tokens):
            index = translate_index(index)
            entry = checkpoint.conversation[index]
            checkpoint.conversation[index] = (entry[0], truncate_to_last_n_tokens(entry[1], num_tokens, gpt4_enc))
        
def take_until(gen, pred, max_num):
    for x in gen:
        if max_num <= 0:
            return
        yield x
        if pred(x):
            max_num -= 1

def is_response(x):
    match x:
        case InsertResponse(_):
            return True
    return False
            
def execute_fixed_prompt(checkpoint, fixed_prompt, args):
    if not inspect.isgeneratorfunction(fixed_prompt):
        raise Exception("Prompt must be generator style")
    transition_rules = itertools.islice(take_until(fixed_prompt(), is_response, args.max_turns), checkpoint.transition_index, args.max_transitions)
    #transition_rules = collect_transition_rules_until_limit(fixed_prompt, response_limit=args.max_turns, total_limit=args.max_entries)
    entries = []

    for transition_rule in transition_rules:
        entries = apply_transition_rule(checkpoint, transition_rule, args)
        if checkpoint.turn_number >= args.max_turns:
            break
        
    return {
        "entries": checkpoint.conversation,
        "statistics": checkpoint,
    }

def log_conversation(checkpoint, log_file_name):
    entries = checkpoint.conversation
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"Conversation started at: {datetime.datetime.now()}\n")
        log_file.write(f"Turn number: {checkpoint.turn_number}\n")
        log_file.write("----\n")

        for i, entry in enumerate(entries):
            speaker, message = entry
            log_file.write(f"Entry {i+1}/{len(entries)} - {speaker}: {message}\n\n")

        log_file.write("----\n")
        log_file.write("Conversation ended.\n")
        log_file.write("\n")

def log_conversation_emily(checkpoint, log_file_name):
    entries = checkpoint.conversation
    args = checkpoint.args
    temperature = 0
    with open(log_file_name, 'a', newline='\r\n') as f:
        f.write(f'model:\n{args.model}\n\n')
        f.write(f'temperature:\n{temperature}\n\n')
        f.write(f'system_message:\n{entries[0][1]}\n\n')
        for index, entry in enumerate(entries[1:-1]):
            speaker, message = entry
            f.write(f'prompt {index + 1} of {len(entries)-2}:\n{message}\n\n')
        f.write(f'response:\n{entries[-1][1]}\n\n')
        f.write('-'*100 + '\n'*11)
        

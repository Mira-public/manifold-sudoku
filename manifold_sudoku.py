#import openai
import json
import time
import os
import requests
import datetime
from dataclasses import dataclass
import re
from sudoku import Sudoku

def convert_pairs_to_openai(entries):
    formatted_messages = [{"role": role, "content": content} for role, content in entries]
    return formatted_messages
import json
import requests

def openai_api_key():
    return os.environ.get("OPENAI_API_KEY")

def find_solved_sudoku(text):
    pattern = r"([1-9][0\-_|\+\s\n]*?){81}"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None

def condense_sudoku(text):
    return ''.join(char for char in text if char.isdigit())
    

@dataclass
class Checkpoint:
    args = None
    turn_number: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    conversation = []
    
    def total_cost(self, model):
        if model == "gpt-3.5-turbo":
            return 0.002 * self.total_tokens / 1000. # $0.002 / 1k tokens
        elif model == "gpt-4":
            return 0.03 * self.prompt_tokens / 1000. + 0.06 * self.output_tokens / 1000.
        
    def save(self):
        checkpoint = {
            "args": vars(self.args),
            "turn_number": self.turn_number,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "conversation": self.conversation,
        }
        with open(self.args.checkpoint, 'w') as f:
            return json.dump(checkpoint, f)

    def load(filename):
        with open(filename, 'r') as f:
            checkpoint = json.load(f)
        ckpt = Checkpoint()
        ckpt.args = checkpoint["args"]
        ckpt.turn_number = checkpoint["turn_number"]
        ckpt.prompt_tokens = checkpoint["prompt_tokens"]
        ckpt.output_tokens = checkpoint["output_tokens"]
        ckpt.total_tokens = checkpoint["total_tokens"]
        ckpt.conversation = checkpoint["conversation"]
        return ckpt
    
def solve_puzzle(puzzle_string):
    board = string_to_list_of_lists(puzzle_string)
    sudoku = Sudoku(3,3,board=board)
    try:
        solved_board = sudoku.solve(raising=True).board
        #print(solved_board)
        return solved_board
    except:
        return None
    
    
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

def run_gpt_4(entries, args, statistics):
    if args.model == 'mock':
        return "mock GPT-4 string" # Use to test without hitting API
    print(f"About to run {args.model}")
    print(f"{len(entries)} Entries: {entries}")
    start_time = time.time()
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     max_tokens=2048,
    #     n=1,
    #     temperature=0,
    #     messages=convert_pairs_to_openai(entries)
    #)
    openai_entries = convert_pairs_to_openai(entries)
    response = openai_chat_completion(openai_entries, args)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for {args.model} call: {elapsed_time:.2f} seconds. {response}")
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

def apply_transition_rule(checkpoint, transition_rule, args):
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
    if transition_rule[0] == "Remove":
        index = translate_index(transition_rule[1])
        checkpoint.conversation.pop(index)
    elif transition_rule[0] == "Insert":
        index, message = translate_index(transition_rule[1]), transition_rule[2]
        insert(index, "user", message)
    elif transition_rule[0] == "InsertPuzzle":
        index, render_func = translate_index(transition_rule[1]), transition_rule[2]
        rendered_puzzle = render_func(args.puzzle)
        insert(index, "user", rendered_puzzle)
    elif transition_rule[0] == "InsertResponse":
        index = translate_index(transition_rule[1])
        response = run_gpt_4(checkpoint.conversation, args, checkpoint)
        insert(index, "assistant", response)
        checkpoint.save() # Long-running API call
        log_conversation(checkpoint.conversation, args.output)
        potential_solution = find_solved_sudoku(response)
        if potential_solution and args.stop_if_solved_puzzle_detected:
            print(f"Found a potential solved Sudoku puzzle:\n{potential_solution}")
            condensed = condense_sudoku(potential_solution)
            solution = solve_puzzle(condensed)
            if solution:
                print("And it was solved: " + str(solution))
                exit()
            else:
                print("But this solution is invalid")
                exit()

def execute_fixed_prompt(checkpoint, fixed_prompt, args):
    transition_rules = collect_transition_rules_until_limit(fixed_prompt, response_limit=args.max_turns, total_limit=args.max_entries)
    entries = []

    for transition_rule in transition_rules:
        entries = apply_transition_rule(checkpoint, transition_rule, args)
        
    return {
        "entries": checkpoint.conversation,
        "statistics": checkpoint,
    }

def log_conversation(entries, log_file_name):
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"Conversation started at: {datetime.datetime.now()}\n")
        log_file.write("----\n")

        for entry in entries:
            speaker, message = entry
            log_file.write(f"{speaker}: {message}\n")

        log_file.write("----\n")
        log_file.write("Conversation ended.\n")
        log_file.write("\n")

# Replace these with actual easy Sudoku puzzles
puzzles = [
    "075400001032000095000125080500230004208510000410008652300050920906040030000309508",
    #"<puzzle_2>",
    #"<puzzle_3>",
    #"<puzzle_4>",
    #"<puzzle_5>",
]

logs = []

# for puzzle in puzzles:
#     log = execute_fixed_prompt(puzzle, fixed_prompt)
#     logs.append(log)

# Validate logs
# valid_logs = [validate_log(log) for log in logs]
# print(f"Validation results: {valid_logs}")

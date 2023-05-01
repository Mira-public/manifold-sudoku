#import openai
import json
import time
import os
import requests
import datetime

MODEL = "gpt-3.5-turbo"

PROMPT_TOKENS = 0
OUTPUT_TOKENS = 0
TOTAL_TOKENS = 0

def convert_pairs_to_openai(entries):
    formatted_messages = [{"role": role, "content": content} for role, content in entries]
    return formatted_messages
import json
import requests

def openai_api_key():
    return os.environ.get("OPENAI_API_KEY")

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
            if attempt < max_retries - 1:
                print("Request failed. Sleeping and then retrying")
                sleep(2 ** attempt)  # Exponential backoff
            else:
                raise Exception(f"OpenAI API request failed after {max_retries} attempts with status code {response.status_code}: {response.text}")

def run_gpt_4(entries, args):
    global PROMPT_TOKENS
    global OUTPUT_TOKENS
    global TOTAL_TOKENS
    #return "mock GPT-4 string" # Use to test without hitting API
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
    TOTAL_TOKENS += response["usage"]["total_tokens"]
    OUTPUT_TOKENS += response["usage"]["completion_tokens"]
    PROMPT_TOKENS += response["usage"]["prompt_tokens"]
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

def apply_transition_rule(entries, transition_rule, args):
    def translate_index(index):
        if index < 0:
            return index + len(entries)+1
        else:
            return index
    def insert(index, role, message):
        index = translate_index(index)
        entries.insert(index, (role, message))
    def remove(index):
        entries.pop(index)
    if transition_rule[0] == "Remove":
        index = translate_index(transition_rule[1])
        entries.pop(index)
    elif transition_rule[0] == "Insert":
        index, message = translate_index(transition_rule[1]), transition_rule[2]
        insert(index, "user", message)
    elif transition_rule[0] == "InsertPuzzle":
        index, render_func = translate_index(transition_rule[1]), transition_rule[2]
        rendered_puzzle = render_func(args.puzzle)
        insert(index, "user", rendered_puzzle)
    elif transition_rule[0] == "InsertResponse":
        index = translate_index(transition_rule[1])
        response = run_gpt_4(entries, args)
        insert(index, "assistant", response)
        log_conversation(entries, args.output)
    return entries

def execute_fixed_prompt(fixed_prompt, args):
    transition_rules = collect_transition_rules_until_limit(fixed_prompt, response_limit=args.max_turns, total_limit=args.max_entries)
    entries = []

    for transition_rule in transition_rules:
        entries = apply_transition_rule(entries, transition_rule, args)
        
    return {
        "entries": entries,
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "total_tokens": TOTAL_TOKENS,
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

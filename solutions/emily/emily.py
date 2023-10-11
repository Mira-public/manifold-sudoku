# $ PYTHONPATH=. python solutions/emily/emily.py

from manifold_sudoku import string_to_list_of_lists, Insert, InsertResponse, Truncate, Remove, InsertPuzzle
from pathlib import Path

def emily_representation(sudoku):
    sudoku_grid = string_to_list_of_lists(sudoku)
    row_names = ['first', 'second', 'third', 'fourth', 
                 'fifth', 'sixth', 'seventh', 'eighth', 'ninth']
    
    output_string = "<output>\n"
    for index, row in enumerate(sudoku_grid):
        # Generate the row name
        row_str = f"{row_names[index]}_row: "
        
        # Generate the row content
        row_content = ','.join(map(str, row))
        row_content = f"[{row_content}]"

        # Append to the output string
        output_string += f"{row_str}{row_content}\n"

    output_string += "</output>"
    return output_string
# Example usage:
# sudoku_grid = [
#     [0,0,0,3,9,4,6,5,0],
#     [0,6,0,0,0,0,0,0,3],
#     [0,0,8,1,5,0,0,0,0],
#     [0,3,9,0,0,7,0,0,0],
#     [4,5,7,0,0,2,0,6,0],
#     [8,0,0,9,0,0,0,1,4],
#     [0,0,0,0,0,0,0,8,0],
#     [9,0,0,0,6,1,0,0,0],
#     [0,1,5,2,8,0,0,4,6]
# ]

#output_string = emily_representation(sudoku_grid)
#print(output_string)

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

script_dir = Path(__file__).resolve().parent

prompt_a = read_file(script_dir / "prompt-A.txt")
prompt_b = read_file(script_dir / "prompt-B.txt")

from .prompt_system import system_message_A, system_message_B, prompt_B_pre_message

def emily_prompt_1():
    yield InsertPuzzle(0, emily_representation)
    while True:
        # Prompt A
        yield Insert(0, system_message_A, tag="system")
        yield Truncate(1, 210) # Truncate last results
        yield Insert(2, "Awaiting instructions.", tag="assistant")
        yield Insert(3, prompt_a)
        yield InsertResponse(-1)

        # Clear everything except the puzzle and response.
        yield Remove(0)
        yield Remove(1)
        yield Remove(1)

        # Prompt B
        yield Insert(0, system_message_B, tag="system")
        yield Insert(1, prompt_B_pre_message)
        yield Insert(2, "Awaiting instructions.", tag="assistant")
        # yield puzzle from Prompt A
        yield Insert(4, "Awaiting instructions.", tag="assistant")
        yield Truncate(5, 300) # <response from previous message>
        yield Insert(6, "Awaiting instructions.", tag="assistant")
        yield Insert(7, prompt_b)
        yield InsertResponse(-1)

        # Clear everything except the response
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)
        yield Remove(0)

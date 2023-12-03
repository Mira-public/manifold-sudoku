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

# Prompt 2
# prompt_a = read_file(script_dir / "two" / "prompt-A.txt")
# prompt_b = read_file(script_dir / "two" / "prompt-B.txt")

# from .two.system_prompts import system_message_A, system_message_B, prompt_B_pre_message

# Prompt 3
prompt_a = read_file(script_dir / "three" / "prompt_A.txt")
prompt_b = read_file(script_dir / "three" / "prompt_B.txt")
system_message_A = read_file(script_dir / "three" / "system_message_prompt_A.txt")
system_message_B = read_file(script_dir / "three" / "system_message_prompt_B.txt")
prompt_B_pre_message = read_file(script_dir / "three" / "prompt_B_pre_message.txt")

def emily_prompt_2():
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
        yield Truncate(5, 400) # <response from previous message>
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


# https://github.com/iamthemathgirlnow/sudoku-challenge-solution/blob/main/version-3/sudoku_tools.py
def find_zeros_in_row_list(row_list):
    zero_indices = []
    for i in range(9):
        zero_indices.append([])
        for j in range(9):
            if row_list[i][j] == 0:
                zero_indices[i].append(j)
    return zero_indices
def rows_to_columns(row_list):
    return [[row_list[j][i] for j in range(9)] for i in range(9)]

def rows_to_squares(row_list):
    return [[row_list[3 * (i // 3) + j // 3][3 * (i % 3) + j % 3] for j in range(9)] for i in range(9)]


def candidates_from_group(group):
    all_digits = [1,2,3,4,5,6,7,8,9]
    return [digit for digit in all_digits if digit not in group]

def all_row_candidates(row_list):    
    candidates_list = []
    for i in range(9):
        candidates_list.append(candidates_from_group(row_list[i]))
    return candidates_list

def all_column_candidates(column_list):
    candidates_list = []
    for i in range(9):
        candidates_list.append(candidates_from_group(column_list[i]))
    return candidates_list

def all_square_candidates(square_list):
    candidates_list = []
    for i in range(9):
        candidates_list.append(candidates_from_group(square_list[i]))
    return candidates_list

def all_candidates(row_list):
    return all_row_candidates(row_list), all_column_candidates(rows_to_columns(row_list)), all_square_candidates(rows_to_squares(row_list))

def string_to_list(puzzle_string):
    return [[int(puzzle_string[i * 9 + j]) for j in range(9)] for i in range(9)]

def list_to_string(puzzle_list):
    return ''.join([str(puzzle_list[i][j]) for i in range(9) for j in range(9)])

def identical_puzzles(puzzle_list_1, puzzle_list_2):
    return list_to_string(puzzle_list_1) == list_to_string(puzzle_list_2)

def list_to_string_list(number_list):
    string_list = []
    for number in number_list:
        string_list.append(str(number))
    string_list = '[' + ','.join(string_list) + ']'
    return string_list


def raw_list_to_list(string_input_sudoku):
    split_list = string_input_sudoku.strip().split('\n')
    if len(split_list) == 11:
        split_list = split_list[1:len(split_list) - 1]
    row_names = []
    rows = []
    for row in split_list:
        row_names.append(row[:row.index(':')])
        rows.append(row[row.index(' ') + 1:])
    
    rows_as_strings = rows.copy()
    rows_as_lists = []
    # convert string of a list to an actual list. For example '[1,2,3,4,5,6,7,8,9]' becomes [1,2,3,4,5,6,7,8,9]
    for row in rows:
        numbers_in_row = row.strip().strip('[]').split(',')
        rows_as_lists.append([int(number) for number in numbers_in_row])

    return row_names, rows_as_lists, rows_as_strings


def display_list(list_to_display):
    for i in range(len(list_to_display)):
        print(list_to_display[i])
    print()

def display_multiple_lists(*args):
    for i in range(len(args)):
        if isinstance(args[i], list):
            display_list(args[i])

def display_full_puzzle_list(puzzle_list):
    print()
    display_list(puzzle_list)
    print()
    display_list(rows_to_columns(puzzle_list))
    print()
    display_list(rows_to_squares(puzzle_list))
    print()


def all_cell_candidates(row_list):
    row_candidates, column_candidates, square_candidates = all_candidates(row_list)
    cell_candidates = []
    for i in range(9):
        cell_candidates.append([])
        for j in range(9):
            if row_list[i][j] == 0:
                cell_candidates[i].append(sorted(list(set(row_candidates[i]) & set(column_candidates[j]) & set(square_candidates[3 * (i // 3) + j // 3]))))
            else:
                cell_candidates[i].append([])
    return cell_candidates


def move_top_row_to_bottom(input_str):
    row_names, row_lists, row_lists_as_strings = raw_list_to_list(input_str)
    # row_names.append(row_names.pop(0))
    row_lists.append(row_lists.pop(0))
    row_lists_as_strings.append(row_lists_as_strings.pop(0))
    # return as input_str
    return '\n'.join([row_names[i] + ': ' + row_lists_as_strings[i] for i in range(len(row_names))]) + '\n'

def move_top_three_rows_to_bottom(input_str):
    rotated_input_str = move_top_row_to_bottom(input_str)
    rotated_input_str = move_top_row_to_bottom(rotated_input_str)
    rotated_input_str = move_top_row_to_bottom(rotated_input_str)
    return rotated_input_str

# Simulating the results of running the current method
def emily_simulate_run(puzzle, print_enabled=None, max_turns=None, max_cells_checked_per_turn=None, max_cells_updated_per_turn=None):
    from math import floor
    row_list = string_to_list_of_lists(puzzle)
    print_enabled = print_enabled if print_enabled is not None else True
    max_turns = floor(max_turns/2) if max_turns is not None else 25
    max_cells_checked_per_turn = max_cells_checked_per_turn if max_cells_checked_per_turn is not None else 20
    max_cells_updated_per_turn = max_cells_updated_per_turn if max_cells_updated_per_turn is not None else 6

    no_cells_found_count = 0
    if print_enabled:
        print()
    for turn in range(1, max_turns+1):
        zero_indices = find_zeros_in_row_list(row_list)
        capped_zero_count = min(sum([len(zero_indices[i]) for i in range(9)]), max_cells_checked_per_turn)
        if capped_zero_count == 0:
            if print_enabled:
                print(f"Sudoku Solved!")
                print(f"Turn {(turn-1)*2}")
            return row_list, (turn-1)*2, "Solved", 1

        empty_cells_checked = 0
        cells_to_update = []
        for i in range(9):
            for j in range(9):
                if empty_cells_checked >= capped_zero_count:
                    break
                if row_list[i][j] == 0:
                    empty_cells_checked += 1
                if row_list[i][j] == 0 and len(all_cell_candidates(row_list)[i][j]) == 1:
                    cells_to_update.append([i,j,all_cell_candidates(row_list)[i][j][0]])

        if print_enabled:
            print(f"Turn {turn*2}")
            print(cells_to_update)

        cells_to_update = cells_to_update[:max_cells_updated_per_turn]
        for cell in cells_to_update:
            row_list[cell[0]][cell[1]] = cell[2]

        if (len(cells_to_update) > 0):
            if print_enabled:
                display_list(row_list)
        else:
            if print_enabled:
                print("No cells with only one candidate found.\n")
            no_cells_found_count += 1
        
        row_list = row_list[3:] + row_list[:3]

        if no_cells_found_count >= 3:
            if print_enabled:
                print("No cells with only one candidate found for all three turns in one cycle, stopping search.")
            return row_list, turn*2, "No Single Candidate Cells", -1

        if turn % 3 == 0:
            no_cells_found_count = 0
        
        zero_indices = find_zeros_in_row_list(row_list)
        capped_zero_count = min(sum([len(zero_indices[i]) for i in range(9)]), max_cells_checked_per_turn)
        if capped_zero_count == 0:
            if print_enabled:
                print(f"Sudoku Solved!")
                print(f"Turn {turn*2}")
            return row_list, turn*2, "Solved", 1
        
    return row_list, turn*2, "Unsolved: Max Turns Reached", -2

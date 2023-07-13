from manifold_sudoku import string_to_2d_representation_no_bars

def joshua_prompt_1(n):
    if n == 0:
        return ("InsertPuzzle", -1, string_to_2d_representation_no_bars)
    elif n == 1:
        return ("InsertResponse", -1)
    else:
        raise Exception("maximum of 1 turn for this prompt")

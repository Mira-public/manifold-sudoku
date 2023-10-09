from manifold_sudoku import string_to_2d_representation_no_bars, InsertPuzzle, InsertResponse

def joshua_prompt_1():
    yield InsertPuzzle(-1, string_to_2d_representation_no_bars)
    yield InsertResponse(-1)


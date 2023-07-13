from manifold_sudoku import string_to_list_of_lists, string_to_visual_representation

def catnee_representation(puzzle):
    rows = 'ABCDEFGHI'
    cols = '123456789'

    puzzle = string_to_list_of_lists(puzzle)
    
    def row_repr(row):
        numbers = [puzzle[row][col] for col in range(9) if puzzle[row][col] != 0]
        return f"In row {rows[row]}, we have {', '.join(map(str, numbers))}."
    
    def col_repr(col):
        numbers = [puzzle[row][col] for row in range(9) if puzzle[row][col] != 0]
        return f"In column {cols[col]}, we have {', '.join(map(str, numbers))}."
    
    def square_repr(square_row, square_col):
        numbers = [puzzle[row][col] for row in range(3 * square_row, 3 * (square_row + 1))
                   for col in range(3 * square_col, 3 * (square_col + 1)) if puzzle[row][col] != 0]
        square_str = "\n".join([" ".join([str(puzzle[row][col]) if puzzle[row][col] != 0 else "_" 
                                          for col in range(3 * square_col, 3 * (square_col + 1))])
                                for row in range(3 * square_row, 3 * (square_row + 1))])
        return f"{rows[3 * square_row]}{cols[3 * square_col]} to {rows[3 * (square_row + 1) - 1]}{cols[3 * (square_col + 1) - 1]}:\n{square_str}\nNumbers present: {', '.join(map(str, numbers))}"
    
    row_reprs = "\n".join([row_repr(row) for row in range(9)])
    col_reprs = "\n".join([col_repr(col) for col in range(9)])
    square_reprs = "\n\n".join([square_repr(square_row, square_col) for square_row in range(3) for square_col in range(3)])
    
    return f"Rows:\n{row_reprs}\n\nColumns:\n{col_reprs}\n\nSquares:\n{square_reprs}"


def catnee_prompt_1(n):
    if n == 0:
        return ("Insert", -1, """Find all cells that have only one candidate in this Sudoku puzzle in one shot like a champ without any mistakes and list all of them

Then check your findings step by step carefully, analyzing every proposed cell by writing existing numbers in the same row, column and square of a proposed cell. don't write your candidate number first, make a guess with a cell, check existing numbers and only then write a complete list of candidates, and only then check if there are more than one

If they really are cells with one candidates, fill them by writing new state of the puzzle

(optional) Point to the mistakes if there are any""")
    elif n == 1:
        return ("InsertPuzzle", -1, string_to_visual_representation)
    elif n == 2:
        return ("Insert", -1, """Here are breakdown of squares:
upper-left square contains cells: c(A,A) c(A,B) c(A,C) c(B,A) c(B,B) c(B,C) c(C,A) c(C,B) c(C,C)
upper-center square contains cells: c(A,D) c(A,E) c(A,F) c(B,D) c(B,E) c(B,F) c(C,D) c(C,E) c(C,F)
upper-right square contains cells: c(A,G) c(A,H) c(A,I) c(B,G) c(B,H) c(B,I) c(C,G) c(C,H) c(C,I)
middle-left square contains cells: c(D,A) c(D,B) c(D,C) c(E,A) c(E,B) c(E,C) c(F,A) c(F,B) c(F,C)
middle-center square contains cells: c(D,D) c(D,E) c(D,F) c(E,D) c(E,E) c(E,F) c(F,D) c(F,E) c(F,F)
middle-right square contains cells: c(D,G) c(D,H) c(D,I) c(E,G) c(E,H) c(E,I) c(F,G) c(F,H) c(F,I)
bottom-left square contains cells: c(G,A) c(G,B) c(G,C) c(H,A) c(H,B) c(H,C) c(I,A) c(I,B) c(I,C)
bottom-center square contains cells: c(G,D) c(G,E) c(G,F) c(H,D) c(H,E) c(H,F) c(I,D) c(I,E) c(I,F)
bottom-right square contains cells: c(G,G) c(G,H) c(G,I) c(H,G) c(H,H) c(H,I) c(I,G) c(I,H) c(I,I)

Here are analysis by sectors:\n\n""")
    elif n == 3:
        return ("InsertPuzzle", -1, catnee_representation)
    elif n == 4:
        return ("Insert", -1, """Example of YOUR work:
–example--
I think Cell (F,E) and Cell (D, A) are good candidates, lets analyze them:

Cell (F, E): In row F, we have 4, 1, 6, and 7. In column E, we have 9, 7, and 6. In the middle-center square, we have 5, 9, 3, and 6. There are no: 2 and 8.

Cell (D, A): In row D, we have 7, 5, 9, 6, and 1. In column A, we have 2, 8, 9, 5, 4, 7, and 1. In the middle-left square, we have 7, 5, 6, 4, and 1. There are no: 3.

Updated puzzle state:
2 1 _ | _ _ _ | 4 8 7
8 _ _ | 3 _ 2 | _ 9 1
9 _ 5 | _ 7 1 | _ _ _
------+------+------
3 _ 7 | 5 9 _ | 6 1 _
5 6 _ | _ _ 3 | _ _ 2
4 _ 1 | 6 _ _ | 7 _ _
------+------+------
_ 3 9 | _ _ 7 | _ _ _
7 _ _ | 1 _ _ | _ 2 6
1 _ _ | _ 6 5 | _ _ 9

Updated sectors states:
In row D, we have 7, 5, 9, 6, 3, and 1. In column A, we have 2, 3, 8, 9, 5, 4, 7, and 1. In the middle-left square, we have 7, 5, 6, 4, 3, and 1.
--example end--

I repeat, do not write your candidate numbers first, make a guess about a cell, check existing numbers and only then write a complete list of candidates, and only then check if there are more than one

repeat these steps until puzzle is solved

don't apologize or say any of that corporate bullshit""")
    elif n == 5:
        return ("Insert", -1, "continue")
    elif n == 6:
        return ("InsertResponse", -1)
    elif n % 4 == 3:
        return ("Insert", -1, "continue")
    elif n % 4 == 0:
        return ("InsertResponse", -1)
    elif n % 4 == 1:
        return ("Remove", 5) # Clear out the old continue message
    elif n % 4 == 2:
        return ("Remove", 5) # Clear out the old ChatGPT message

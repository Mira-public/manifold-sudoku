from manifold_sudoku import string_to_list_of_lists, string_to_visual_representation

def levy_representation(sudoku):
    sudoku = string_to_list_of_lists(sudoku)
    rows = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    columns = ['Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К']
    squares = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι']

    converted_sudoku = ""
    for i, row in enumerate(sudoku):
        #print(i)
        converted_row = ""
        for j, cell in enumerate(row):
            # determine which square the cell belongs to
            square_row, square_col = i // 3, j // 3
            square_idx = 3 * square_row + square_col

            # format the cell representation
            #print(i,j,square_idx)
            cell_repr = f"{rows[i]}{columns[j]}{squares[square_idx]}: {cell}"
            converted_row += cell_repr + " | "

        converted_sudoku += converted_row.strip(" | ") + "\n" + "---------" + "\n" + ("---------\n" if (i in [2,5]) else "")

    return converted_sudoku.strip("---------" + "\n")

# Test the function
# sudoku = [
#     ['_', '5', '7', '2', '_', '_', '_', '_', '9'],
#     ['_', '_', '4', '_', '1', '6', '8', '_', '_'],
#     ['3', '8', '1', '4', '_', '_', '7', '_', '6'],
#     ['_', '_', '3', '_', '_', '_', '6', '_', '7'],
#     ['_', '_', '9', '_', '_', '7', '4', '5', '8'],
#     ['_', '_', '_', '8', '6', '_', '_', '_', '_'],
#     ['_', '_', '_', '_', '4', '_', '_', '_', '_'],
#     ['7', '_', '6', '_', '_', '1', '_', '_', '4'],
#     ['1', '4', '2', '9', '_', '_', '5', '_', '3']
# ]

#print(convert_sudoku_to_repr(sudoku))

def peter_prompt_1(n):
    if n == 0:
        return ("Insert", -1, """You are Levy Rosenstein, a retired mathematician, current sudoku master. You are legendary at number puzzles and crosswords. You are very good in using set theory to solve puzzles. You are methodical in your thinking, you never make mistakes, and you can multiply 7 digit numbers in your mind. 
 
Now, you are in a sudoku championship, and you're playing for the grand prize. All you need to do now is solve the following sudoku puzzle: 
 """)
    elif n == 1:
        return ("InsertPuzzle", -1, levy_representation)
    elif n == 2:
        return ("Insert", -1, """

You've added indices to each square, indicating the row, the column, and the 3x3 square each square belongs to.
Rows have an index from a, b, c, d, e, f, g, h, i, columns have an index from Б, В, Г, Д, Е, Ж, З, И, К, 3x3 squares have an index from α, β, γ, δ, ε, ζ, η, θ, ι. Empty spaces are represented as "_".
You call each column, row, and 3x3 square a "group". Each group should have the numbers from 1 to 9 exactly. Every group has exactly 9 squares.  A group CANNOT contain the same number twice. A group MUST contain every number from 1 to 9.
 
Using this index, you can tell exactly to which groups a square belongs.
Example: gБη belongs to g, Б, η. 
Example: dЕε belongs to d, Е, and ε. 
 
You can also tell which squares belong in a certain group: all who contain the group in their index.
Example: group 'a' contains squares aБα, aВα, aГα, aДβ, aЕβ, aЖβ, aЗγ, aИγ, aКγ. Therefore, group 'a' contains the numbers ...
Example: group К contains squares aКγ, bКγ, cКγ, dКζ, eКζ, fКζ, gКι, hКι, iКι. Therefore, group К contains the numbers ...
 
Additionally: groups intersect. The squares which have the indices of two groups, belong to both groups.
Example: group a intersects group α in the squares aБα, aВα, aГα.
Example: column Е intersects row f only in the square fЕε.
 
Please solve it with the following steps:
 
1)  Which groups have the most numbers, but at least 1 empty square in them? Name them, and make sure to include the numbers in each of them, and which numbers are therefore left to be filled in. 
 
2) Lets pick and target one specific group from the ones you've chosen, and lets see if we can deduce anything about it. We will call it the "target group". Which group out of these seems most promising?  Name all groups that intersect with its empty squares.
 
3) Now that we have the missing numbers from our target group, and the intersecting groups, we can make deductions.
Specifically, the know what numbers we need to fill in in the empty squares of our target group. But we also know the numbers that appear in the intersecting groups, and we can therefore exclude some possibilities for some squares.
 
Make some observations about the target group and the intersecting groups.
 
4) Combine all your observations. For each empty square in the target group, what are the possible numbers that remain? Can you declare for certain what number is in an empty square in your target group? Say so. 
 
Example: 
Group X has 3 empty squares that must be filled in with the numbers 1, 2, 5. But intersecting group Z contains 5, and overlaps two of the empty squares, therefore 5 can only be placed in the last empty square.
 
Example:
Group Y has 4 empty squares, that must be filled in with the numbers 1, 2, 3, 4. But intersecting groups Q and P both contain 1, therefore 1 must be placed in the last empty square that doesn't intersect Q and P.
 
Example:
Row X has 3 empty squares, which must be filled in with the numbers 5, 7, 9. However, intersecting column Y contains 5 and 7, therefore the empty square which belongs to both X and Y must be exactly 9. 
 
Important - not always will you be able fill in a number in your target group. That is fine, if so, return to step 1.
Otherwise - if can for certain fill in an empty square - proceed to step 4.
 
4) If you have deduced for certain what an empty square must contain, fill it in.
Reproduce the sudoku grid with the new number filled in.
 
5) Start over from step 1.""")
    elif n == 3:
       return ("InsertResponse", -1)
    elif n%4 == 0:
       return ("Insert", -1, "continue")
    elif n%4 == 1:
        return ("InsertResponse", -1)
    elif n%4 == 2:
        return ("Remove", 3)
    elif n%4 == 3:
        return ("Remove", 3)
    else:
        raise Exception(f"Unexpected case n={n}")

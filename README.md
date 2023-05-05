# manifold-sudoku

Judging script used for the [M20k Sudoku Challenge](https://manifold.markets/Mira/will-a-prompt-that-enables-gpt4-to?r=TWlyYQ)

Instructions:
1. Install python. Should be no extra libraries needed.
2. Set OPENAI_API_KEY environment variable
3. `python3 main.py run-prompt --help` to see every option, or open `run.sh` to see an example invocation.
4. See `solutions/catnee.py` to see an example prompt.

Puzzles should be provided in the format "604000053000000092080400710800071500090305000025000070036014000100702000500806024" which are the numbers left-to-right, top-to-bottom with 0 representing blank:
```
6 _ 4 | _ _ _ | _ 5 3
_ _ _ | _ _ _ | _ 9 2
_ 8 _ | 4 _ _ | 7 1 _
-+------+------+------
8 _ _ | _ 7 1 | 5 _ _
_ 9 _ | 3 _ 5 | _ _ _
_ 2 5 | _ _ _ | _ 7 _
-+------+------+------
_ 3 6 | _ 1 4 | _ _ _
1 _ _ | 7 _ 2 | _ _ _
5 _ _ | 8 _ 6 | _ 2 4
```

Script will generate a "checkpoint file" ending in ".ckpt" in case you want to restart a run(OpenAI API calls can be very slow and expensive, so this was worthwhile to implement), and save to a log file that includes the puzzle string.

Prompts must be a function taking a turn number to one of:
```
("Insert", -1, "some message") // Insert a message at the end
("InsertResponse", -1) // Insert GPT response at the end
("Remove", -1) // Delete the last message from context
("InsertPuzzle", -1, string_to_visual_representation) // Insert a rendered version of the puzzle, using the given function
```

GPT-4 can be used to convert the market description and an example prompt to Python code that generates this format(the `solutions/catnee.py` example was mostly GPT-generated).


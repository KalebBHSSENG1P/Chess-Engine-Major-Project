# AI USAGE DOCUMENTATION (MICROSOFT COPILOT)
## 1. Request a high‑level code review of the uploaded engine
Please review the uploaded SmartMoveFinder.py file and provide a concise technical assessment. Identify any functional bugs, incorrect assumptions, or likely runtime errors. For each issue, include a short explanation, the exact file location (function and approximate line), and a minimal code change that would fix it.

## 2. Fix pin detection implementation
Provide a corrected implementation of the pin detection routine for _move_quality in SmartMoveFinder.py. The fix must:
- Detect pins using the board state before a move is applied.
- Identify the pattern attacker → moved piece → higher‑value friendly piece.
- Penalize only when the moved piece is the pinned piece.
- Return a drop‑in code block that uses the existing local variables and matches the project’s coding style.

## 3. Integrate fix inside existing try block and use existing variables
Rewrite the pin detection fix so it uses the variable names already present in _move_quality and is placed inside the existing try: block. Do not introduce helper functions or new abstractions. Preserve the file’s coding conventions and minimal variable footprint.

## 4. Minimise new variables and preserve file style
Produce the pin detection code again but restrict new identifiers to only those that are strictly necessary for loop control. Reuse start_r, start_c, piece, color, and other variables already defined in the file. Ensure the code is stylistically consistent with the repository and contains no helper functions.

## 5. Add extensive inline comments at high density
Annotate the pin detection code with detailed inline comments at approximately a 3:1 comment‑to‑code ratio. Keep the code unchanged; only add comments that explain intent, edge cases, and the reasoning behind each conditional and loop.

## 6. Reduce comment density to lightweight level
Replace the heavy comments with concise inline comments at approximately a 1:3 comment‑to‑code ratio. Maintain clarity while avoiding excessive verbosity. Keep the code unchanged.

## 7. Diagnose why engine prefers sacrificial checks
Analyze the engine’s evaluation and tiebreaker logic and explain why it prefers moves that give check while sacrificing material. Identify the specific lines or heuristics that cause this behavior, quantify their relative impact where possible, and propose minimal, concrete parameter changes to correct the imbalance. Provide verification steps and test positions to confirm the fix.

## 8. Explain why pin logic never triggers and propose fix
Investigate why the existing pin detection logic does not detect real pins in practice. Provide a precise diagnosis listing the root causes, and supply a corrected algorithm that:
- Runs before the move is applied,
- Requires only the moved piece and a higher‑value friendly piece behind it (including the king),
- Uses only existing variables and minimal loop counters.
- Include a short unit test example that demonstrates the corrected behavior.

## 9. Where to apply unit tests in the project
Recommend a test strategy for this repository. Specify which modules and functions should receive pytest or unittest coverage, prioritize tests by impact, and provide example test cases for GameState, Move, and the AI heuristics. Include guidance on CI configuration and which tests to keep fast versus which to run less frequently.
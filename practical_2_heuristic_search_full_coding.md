# Practical 2 — AI-Based Puzzle Solver Using Heuristic Search

## Duration
**2 Hours — Coding Practical**

## Objective

Implement and compare:

- **Hill Climbing**
- **A\* Search**

Use the **8-Puzzle** as the main problem.

---

# 1. Problem: 8-Puzzle

Represent the puzzle as a 3×3 board.

### Initial State

```text
1 2 3
4 5 6
_ 7 8
```

### Goal State

```text
1 2 3
4 5 6
7 8 _
```

The `_` represents the blank space.

### State

A state is one complete arrangement of the 8 tiles and the blank.

### Actions

Move the blank:

- Up
- Down
- Left
- Right

Only legal moves are allowed.

---

# 2. What You Need to Code

Build the solver in these stages:

```text
1. Represent a puzzle state
        ↓
2. Generate valid moves
        ↓
3. Calculate heuristic
        ↓
4. Implement Hill Climbing
        ↓
5. Implement A*
        ↓
6. Compare both algorithms
```

---

# 3. Step 1 — Represent a State

Use a list/tuple.

Example:

```python
state = (
    1, 2, 3,
    4, 5, 6,
    0, 7, 8
)
```

Use `0` for the blank.

Store the goal as:

```python
goal = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)
```

### Task

Write code to:

- Print a state as a 3×3 board.
- Check whether a state is the goal.

---

# 4. Step 2 — Generate Neighboring States

Write a function:

```python
get_neighbors(state)
```

It should:

1. Find the position of `0`.
2. Determine legal moves.
3. Create a new state for each legal move.
4. Return all neighboring states.

### Example

For:

```text
1 2 3
4 5 6
_ 7 8
```

Possible moves are:

```text
UP
RIGHT
```

because the blank cannot move further down or left.

### Task

Test your function and print all possible next states.

---

# 5. Step 3 — Heuristic Function

Use **Manhattan Distance**.

For each tile:

```text
distance =
|current_row - goal_row|
+
|current_col - goal_col|
```

Then add the distances of all tiles.

Do not include the blank (`0`).

### Example

```python
def manhattan_distance(state, goal):
    ...
```

### Task

Test your heuristic on at least three different states.

Example:

```text
State A → h = ?
State B → h = ?
State C → h = ?
```

Verify that a state closer to the goal generally has a smaller heuristic.

---

# 6. Step 4 — Hill Climbing

Implement:

```python
hill_climbing(start, goal)
```

Use this logic:

```text
current = start

repeat:
    if current is goal:
        return solution

    generate neighbors

    evaluate every neighbor using h(n)

    choose the neighbor with the best heuristic

    if no neighbor is better:
        stop

    current = best neighbor
```

For this problem, a **smaller Manhattan distance is better**.

### Important

Hill Climbing is a local search method.

It may:

- reach the goal,
- get stuck,
- or fail to find a solution even when one exists.

---

# 7. Step 5 — Show the Hill-Climbing Path

Your program should print something like:

```text
Hill Climbing

Step 0:
1 2 3
4 5 6
0 7 8

h = 2

Step 1:
1 2 3
4 5 6
7 0 8

h = 1

Step 2:
1 2 3
4 5 6
7 8 0

h = 0

Goal reached!
```

Also display:

```text
States explored: ...
Steps: ...
Execution time: ...
```

---

# 8. Step 6 — A* Search

Implement:

```python
a_star(start, goal)
```

Use:

\[
f(n)=g(n)+h(n)
\]

where:

- `g(n)` = cost from start to current state
- `h(n)` = Manhattan distance estimate
- `f(n)` = estimated total cost

For this practical, every tile movement has cost:

```text
1
```

Therefore:

```text
g(child) = g(parent) + 1
```

---

# 9. A* Data to Maintain

For each state, maintain:

```python
g_score[state]
h_score[state]
f_score[state]
parent[state]
```

You will also need a structure for states that still need to be explored.

A Python priority queue may be used:

```python
import heapq
```

The priority should be based on:

```python
f(n)
```

---

# 10. A* Basic Logic

Use this structure:

```text
Put start into priority queue

while queue is not empty:

    remove state with smallest f(n)

    if it is the goal:
        reconstruct the path
        return solution

    generate neighbors

    for each neighbor:

        calculate new g(n)

        if this is a better path:
            update parent
            calculate h(n)
            calculate f(n)
            add/update neighbor in queue
```

Do not copy a library implementation blindly.

The aim is to understand how `g`, `h`, and `f` control the search.

---

# 11. Reconstruct the Solution Path

Store:

```python
parent[child] = current
```

When the goal is reached:

```text
Goal
 ↑
parent
 ↑
parent
 ↑
Start
```

Write a function:

```python
reconstruct_path(parent, goal)
```

It should return the complete sequence of states from start to goal.

---

# 12. Display A* Search

Print:

```text
A* Search

Step 0:
...

g = ...
h = ...
f = ...

Step 1:
...

g = ...
h = ...
f = ...

...

Goal reached!
```

Also display:

```text
States explored: ...
Path cost: ...
Number of moves: ...
Execution time: ...
```

---

# 13. Main Program

Create a simple main program:

```python
def main():

    start = (
        1, 2, 3,
        4, 5, 6,
        0, 7, 8
    )

    goal = (
        1, 2, 3,
        4, 5, 6,
        7, 8, 0
    )

    print("===== HILL CLIMBING =====")
    hill_climbing(start, goal)

    print("===== A* SEARCH =====")
    a_star(start, goal)


if __name__ == "__main__":
    main()
```

---

# 14. Required Comparison

Run both algorithms on the **same initial state**.

Record:

| Measure | Hill Climbing | A* |
|---|---:|---:|
| Solution found? | | |
| Number of moves | | |
| Path cost | | |
| States explored | | |
| Execution time | | |

### Questions

1. Which algorithm found a solution?
2. Which explored fewer/more states?
3. Did Hill Climbing get stuck?
4. Was the solution from Hill Climbing optimal?
5. What role did the Manhattan heuristic play?
6. Why can A* consider more information than Hill Climbing?

---

# 15. Practical Test Cases

Run at least **three** initial states.

### Test Case 1 — Easy

```text
1 2 3
4 5 6
0 7 8
```

### Test Case 2 — Moderate

```text
1 2 3
5 0 6
4 7 8
```

### Test Case 3 — More Difficult

Use a valid shuffled state.

For each test case, compare both algorithms.

---

# 16. Optional Extension — Maze

If time permits, repeat the same idea with a small maze.

Example:

```text
S 0 0 1 0
1 0 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 0 G
```

Use:

```text
h(n) = Manhattan distance to goal
```

Students can then observe that the same heuristic-search ideas apply to a different problem representation.

---

# 17. Suggested 2-Hour Class Plan

### 0–15 min
Understand the 8-Puzzle and state representation.

### 15–30 min
Code state printing and goal checking.

### 30–50 min
Code neighbor generation.

### 50–65 min
Implement Manhattan Distance.

### 65–85 min
Implement Hill Climbing.

### 85–105 min
Implement A*.

### 105–115 min
Run test cases and collect comparison results.

### 115–120 min
Discuss results and submit output.

---

# 18. Minimum Deliverables

Submit:

```text
1. Python source code
2. Output for 3 test cases
3. Comparison table
4. Short conclusion
```

### Conclusion

Write 4–5 lines explaining:

- Which algorithm performed better on your test cases.
- Whether Hill Climbing always found a solution.
- How the heuristic reduced uninformed/random exploration.
- Why A* was able to balance actual and estimated cost.

---

# 19. What You Should Understand After the Practical

By the end, you should be able to explain:

```text
State
↓
Neighbors
↓
Heuristic h(n)
↓
Hill Climbing:
choose a better neighbor

A*:
g(n) + h(n)
↓
choose the best estimated total cost
```

The goal of this practical is **not just to make the program work**.

You should understand how the heuristic changes the way the search explores the problem.

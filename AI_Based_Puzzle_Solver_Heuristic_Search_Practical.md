# Practical Class: Design an AI-Based Puzzle Solver Using Heuristic Search

## 1. Practical Overview

**Practical Title:** Design an AI-Based Puzzle Solver using Heuristic Search

**Objective:**  
Develop an intelligent solver for a problem such as the **8-Puzzle** using **Hill Climbing** and **A\* Search**. Compare these heuristic search techniques with an uninformed search method such as **Breadth-First Search (BFS)** and observe how heuristic functions improve search efficiency.

---

## 2. Learning Outcomes

By the end of this practical, students should be able to:

- Represent the 8-Puzzle as a search problem.
- Define **initial state**, **goal state**, **actions**, and **path cost**.
- Understand the role of a **heuristic function**.
- Implement Hill Climbing.
- Implement A\* Search.
- Implement an uninformed search method such as BFS.
- Compare search algorithms using metrics such as:
  - Number of states/nodes explored
  - Solution path length
  - Execution time
  - Whether a solution is found
- Explain why an appropriate heuristic can reduce unnecessary exploration.

---

# 3. Problem Statement

Consider the following **8-Puzzle**:

```text
Initial State              Goal State

1  2  3                     1  2  3
5  6  _                     4  5  6
4  7  8                     7  8  _
```

The blank space `_` can move **up, down, left, or right** when the corresponding position is available.

### Task

Develop an AI-based puzzle solver that finds a sequence of moves from the initial state to the goal state using:

1. **BFS** — uninformed search
2. **Hill Climbing** — heuristic search
3. **A\*** — informed/heuristic search

Then compare their performance.

---

# 4. Search Problem Formulation

For the 8-Puzzle:

### Initial State

The starting arrangement of the puzzle.

Example:

```text
1 2 3
5 6 _
4 7 8
```

### Goal State

The arrangement that the algorithm must reach.

```text
1 2 3
4 5 6
7 8 _
```

### Actions

The blank tile can move:

- Up
- Down
- Left
- Right

provided that the move is valid.

### State

A state is one particular arrangement of all 8 numbered tiles and the blank space.

For example:

```text
(1, 2, 3,
 5, 6, 0,
 4, 7, 8)
```

Here, `0` represents the blank.

### Path Cost

Assume every move has a cost of **1**.

Therefore:

```text
g(n) = number of moves made to reach state n
```

---

# 5. Heuristic Functions

A heuristic estimates how close the current state is to the goal.

We will use two common heuristics.

## Heuristic 1: Misplaced Tiles

Count how many tiles are not in their correct positions.

```text
h(n) = number of misplaced tiles
```

Do not count the blank tile.

Example:

```text
Current:                 Goal:

1 2 3                    1 2 3
5 6 _                    4 5 6
4 7 8                    7 8 _

Misplaced tiles:
5, 6, 4, 7, 8

h(n) = 5
```

---

## Heuristic 2: Manhattan Distance

For every tile, calculate how many horizontal and vertical moves it is away from its goal position.

```text
h(n) = sum of Manhattan distances of all tiles
```

For a tile:

```text
Manhattan Distance =
|current row - goal row| +
|current column - goal column|
```

The blank is not included.

### Example

Suppose tile `5` is currently at:

```text
row = 2, column = 1
```

and its goal position is:

```text
row = 2, column = 2
```

Then:

```text
Distance = |2 - 2| + |1 - 2|
         = 1
```

The Manhattan distances of all tiles are added to obtain `h(n)`.

---

# 6. BFS — Uninformed Search

BFS does not use a heuristic.

It explores states level by level:

```text
Depth 0
   ↓
Depth 1
   ↓
Depth 2
   ↓
Depth 3
   ↓
...
```

### Important idea

BFS asks:

> "Which state was generated earliest?"

It does **not** ask:

> "Which state looks closer to the goal?"

For an unweighted puzzle where every move costs 1, BFS can find an optimal solution, but it may explore many unnecessary states.

---

# 7. Hill Climbing

Hill Climbing uses a heuristic to choose the next state.

For every available move:

1. Generate neighboring states.
2. Calculate their heuristic values.
3. Select the neighbor with the lowest `h(n)`.
4. Move to that state.
5. Repeat until the goal is reached.

### Basic idea

```text
Current State
      |
      ↓
Generate neighbors
      |
      ↓
Calculate h(n)
      |
      ↓
Choose best neighbor
      |
      ↓
Repeat
```

### Example

Suppose the current state has three valid neighbors:

| State | h(n) |
|---|---:|
| A | 5 |
| B | 3 |
| C | 4 |

Hill Climbing chooses:

```text
B because h(B) = 3
```

### Problem with Hill Climbing

Hill Climbing can get stuck in:

- Local minima
- Plateaus
- Situations where no neighboring state appears better

Therefore:

> Hill Climbing is not guaranteed to find a solution, even when one exists.

---

# 8. A* Search

A\* combines:

- The actual cost already spent
- The estimated cost remaining

The evaluation function is:

```text
f(n) = g(n) + h(n)
```

Where:

- `g(n)` = cost from the initial state to `n`
- `h(n)` = estimated cost from `n` to the goal
- `f(n)` = estimated total cost of the solution through `n`

### Example

Suppose:

```text
g(n) = 4
h(n) = 3
```

Then:

```text
f(n) = 4 + 3
     = 7
```

A\* generally expands the state with the smallest `f(n)`.

---

# 9. Why A* Is More Intelligent Than BFS

Consider two states:

| State | g(n) | h(n) | f(n) |
|---|---:|---:|---:|
| A | 2 | 8 | 10 |
| B | 4 | 3 | 7 |

BFS does not use `h(n)`.

A\* considers:

```text
f(n) = g(n) + h(n)
```

Therefore it prefers:

```text
B because f(B) = 7
```

This allows A\* to focus its search toward promising states instead of blindly exploring states level by level.

---

# 10. Practical Tasks

## Task 1 — Represent the Puzzle

Represent the puzzle using a suitable Python data structure.

For example:

```python
initial_state = (1, 2, 3,
                 5, 6, 0,
                 4, 7, 8)

goal_state = (1, 2, 3,
              4, 5, 6,
              7, 8, 0)
```

Use `0` for the blank tile.

---

## Task 2 — Generate Valid Moves

Create a function:

```python
get_neighbors(state)
```

The function should:

1. Locate the blank tile.
2. Determine possible movements.
3. Generate all valid neighboring states.
4. Return those states.

For example:

```text
Move blank Up
Move blank Down
Move blank Left
Move blank Right
```

Only valid moves should be generated.

---

## Task 3 — Implement the Misplaced-Tile Heuristic

Create:

```python
def misplaced_tiles(state, goal):
    ...
```

Return the number of misplaced numbered tiles.

---

## Task 4 — Implement Manhattan-Distance Heuristic

Create:

```python
def manhattan_distance(state, goal):
    ...
```

For each numbered tile:

```text
distance =
|current_row - goal_row| +
|current_col - goal_col|
```

Return the sum.

---

## Task 5 — Implement BFS

Create:

```python
def bfs(initial, goal):
    ...
```

The algorithm should:

1. Add the initial state to a queue.
2. Remove a state from the front.
3. Check whether it is the goal.
4. Generate its neighbors.
5. Add unvisited neighbors to the queue.
6. Continue until the goal is found or the queue becomes empty.

Maintain a `visited` set to avoid repeatedly exploring the same state.

---

## Task 6 — Implement Hill Climbing

Create:

```python
def hill_climbing(initial, goal, heuristic):
    ...
```

At every step:

```text
Current State
      ↓
Generate neighbors
      ↓
Calculate h(n)
      ↓
Select best neighbor
      ↓
Is it better than current?
      ↓
Yes → Move
No  → Stop / Local Minimum
```

The implementation should report when it gets stuck.

---

## Task 7 — Implement A*

Create:

```python
def a_star(initial, goal, heuristic):
    ...
```

For every state calculate:

```text
g(n)
h(n)
f(n) = g(n) + h(n)
```

Use a priority queue to select the state with the smallest `f(n)`.

The implementation should reconstruct and display the final solution path.

---

# 11. Suggested A* Data Representation

Each search node can contain:

```text
state
parent
g
h
f
```

For example:

```text
State = current puzzle arrangement
g     = moves from start
h     = heuristic estimate
f     = g + h
parent = previous state
```

This allows the solution path to be reconstructed after reaching the goal.

---

# 12. Displaying the Solution

Do not only print:

```text
Solution Found
```

Display the sequence of puzzle states.

Example:

```text
Initial State

1 2 3
5 6 _
4 7 8

       ↓

1 2 3
5 _ 6
4 7 8

       ↓

1 2 3
_ 5 6
4 7 8

       ↓

1 2 3
4 5 6
7 8 _
```

Also display:

```text
Number of moves: 3
```

---

# 13. Performance Evaluation

Run all three algorithms on the same puzzle.

Record:

| Algorithm | Heuristic | Nodes Explored | Solution Moves | Time | Solved? |
|---|---|---:|---:|---:|---|
| BFS | None | ___ | ___ | ___ | Yes/No |
| Hill Climbing | Misplaced | ___ | ___ | ___ | Yes/No |
| A* | Misplaced | ___ | ___ | ___ | Yes/No |
| A* | Manhattan | ___ | ___ | ___ | Yes/No |

### Important

Use the **same initial state** for every algorithm so that the comparison is fair.

---

# 14. Questions for Analysis

Answer the following after completing the implementation.

### Q1.
Why does BFS not need a heuristic?

### Q2.
Why can Hill Climbing get stuck even when a solution exists?

### Q3.
What is the difference between:

```text
h(n)
```

and

```text
f(n) = g(n) + h(n)
```

?

### Q4.
Which heuristic performed better for the 8-Puzzle?

```text
Misplaced Tiles
OR
Manhattan Distance
```

Give an explanation based on your experimental results.

### Q5.
Why does Manhattan Distance generally provide more information than simply counting misplaced tiles?

### Q6.
Which algorithm explored the fewest states?

### Q7.
Did the algorithm that explored the fewest states always produce the shortest solution?

### Q8.
What happens to search efficiency as the puzzle becomes more difficult?

---

# 15. Challenge Task

Modify the program so that the user can enter an arbitrary 8-Puzzle configuration.

Example:

```text
Enter puzzle:

1 2 3
4 0 6
7 5 8
```

The program should:

1. Validate the input.
2. Check whether the puzzle is solvable.
3. Run BFS.
4. Run Hill Climbing.
5. Run A*.
6. Display the solution path.
7. Display performance metrics.
8. Compare the algorithms.

---

# 16. Advanced Challenge

Implement A\* using both heuristics:

```text
A* + Misplaced Tiles
A* + Manhattan Distance
```

Then determine:

> Which heuristic allows A\* to reach the solution while exploring fewer states?

Also explain why.

---

# 17. Expected Conceptual Comparison

| Feature | BFS | Hill Climbing | A* |
|---|---|---|---|
| Uses heuristic | No | Yes | Yes |
| Uses path cost | Yes | Usually No | Yes |
| Evaluation | `g(n)` | `h(n)` | `g(n)+h(n)` |
| Can get stuck locally | No | Yes | No, with appropriate implementation |
| Complete | Yes for finite branching | No | Yes under standard conditions |
| Optimal | Yes for unit-cost moves | No | Yes with an admissible heuristic |
| Search direction | Uninformed | Greedy/local | Informed |

---

# 18. Key Takeaways

### BFS

```text
Search without knowledge of the goal's direction.
```

### Hill Climbing

```text
Always move toward the state that looks better according to h(n).
```

### A*

```text
Consider both:
"What have I already spent?"
+
"What do I estimate remains?"
```

Therefore:

```text
A* → f(n) = g(n) + h(n)
```

The main purpose of this practical is not only to find a solution, but to **experimentally observe how heuristic information can reduce unnecessary search**.

---

# 19. Submission Requirements

Submit:

- Python source code
- Output screenshots
- Comparison table
- Answers to the analysis questions
- Short conclusion

### Conclusion should discuss

1. Which algorithm was fastest?
2. Which explored the fewest states?
3. Which produced the shortest solution?
4. Which heuristic performed better?
5. Why heuristic information improves search efficiency.


# 9. Complete Python Implementation

```python
from collections import deque
import heapq
import time

INITIAL_STATE = (
    1, 2, 3,
    5, 6, 0,
    4, 7, 8
)

GOAL_STATE = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)


def print_state(state):
    for i in range(0, 9, 3):
        row = state[i:i + 3]
        print(" ".join("_" if x == 0 else str(x) for x in row))
    print()


def get_neighbors(state):
    """Return all valid next states and the move used."""
    neighbors = []
    zero = state.index(0)

    row = zero // 3
    col = zero % 3

    moves = [
        (-1, 0, "Up"),
        (1, 0, "Down"),
        (0, -1, "Left"),
        (0, 1, "Right")
    ]

    for dr, dc, move_name in moves:
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = new_row * 3 + new_col

            new_state = list(state)
            new_state[zero], new_state[new_index] = (
                new_state[new_index],
                new_state[zero]
            )

            neighbors.append((tuple(new_state), move_name))

    return neighbors


def misplaced_tiles(state, goal):
    """Number of numbered tiles in the wrong position."""
    return sum(
        1 for current, target in zip(state, goal)
        if current != 0 and current != target
    )


def manhattan_distance(state, goal):
    """Sum of Manhattan distances of all numbered tiles."""
    distance = 0

    for tile in range(1, 9):
        current_index = state.index(tile)
        goal_index = goal.index(tile)

        current_row, current_col = divmod(current_index, 3)
        goal_row, goal_col = divmod(goal_index, 3)

        distance += (
            abs(current_row - goal_row)
            + abs(current_col - goal_col)
        )

    return distance


def count_inversions(state):
    numbers = [x for x in state if x != 0]
    return sum(
        1
        for i in range(len(numbers))
        for j in range(i + 1, len(numbers))
        if numbers[i] > numbers[j]
    )


def is_solvable(state):
    """For a 3x3 puzzle, an even inversion count is solvable."""
    return count_inversions(state) % 2 == 0


def reconstruct_path(parent, goal):
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    return path[::-1]


def bfs(initial, goal):
    """Breadth-First Search."""
    start = time.perf_counter()

    queue = deque([initial])
    visited = {initial}
    parent = {initial: None}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            return (
                reconstruct_path(parent, goal),
                nodes_expanded,
                time.perf_counter() - start
            )

        for neighbor, _ in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return None, nodes_expanded, time.perf_counter() - start


def hill_climbing(initial, goal, heuristic):
    """Steepest-descent Hill Climbing using h(n)."""
    start = time.perf_counter()

    current = initial
    path = [current]
    visited = {current}
    nodes_expanded = 0

    while current != goal:
        neighbors = [
            (state, move)
            for state, move in get_neighbors(current)
            if state not in visited
        ]

        nodes_expanded += len(neighbors)

        if not neighbors:
            return None, nodes_expanded, time.perf_counter() - start

        best_state, _ = min(
            neighbors,
            key=lambda item: heuristic(item[0], goal)
        )

        current_h = heuristic(current, goal)
        best_h = heuristic(best_state, goal)

        if best_h >= current_h:
            # Local minimum / plateau
            return None, nodes_expanded, time.perf_counter() - start

        current = best_state
        visited.add(current)
        path.append(current)

    return path, nodes_expanded, time.perf_counter() - start


def a_star(initial, goal, heuristic):
    """A* Search using f(n) = g(n) + h(n)."""
    start = time.perf_counter()

    # (f, g, tie_breaker, state)
    priority_queue = []
    counter = 0

    h = heuristic(initial, goal)

    heapq.heappush(
        priority_queue,
        (h, 0, counter, initial)
    )

    g_cost = {initial: 0}
    parent = {initial: None}
    closed = set()
    nodes_expanded = 0

    while priority_queue:
        f, current_g, _, current = heapq.heappop(priority_queue)

        if current in closed:
            continue

        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return (
                reconstruct_path(parent, goal),
                nodes_expanded,
                time.perf_counter() - start
            )

        for neighbor, _ in get_neighbors(current):
            if neighbor in closed:
                continue

            new_g = current_g + 1

            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                h = heuristic(neighbor, goal)
                f = new_g + h

                counter += 1

                heapq.heappush(
                    priority_queue,
                    (f, new_g, counter, neighbor)
                )

                parent[neighbor] = current

    return None, nodes_expanded, time.perf_counter() - start


def print_solution(path):
    if path is None:
        print("No solution found.")
        return

    for step, state in enumerate(path):
        print(f"Step {step}:")
        print_state(state)


def run_algorithm(name, algorithm, initial, goal, heuristic=None):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if heuristic is None:
        path, nodes, elapsed = algorithm(initial, goal)
    else:
        path, nodes, elapsed = algorithm(initial, goal, heuristic)

    solved = path is not None

    print("Solved:", "Yes" if solved else "No")
    print("Nodes expanded:", nodes)
    print(f"Execution time: {elapsed:.6f} seconds")

    if solved:
        print("Solution moves:", len(path) - 1)
        print_solution(path)

    return {
        "algorithm": name,
        "path": path,
        "nodes": nodes,
        "time": elapsed
    }


def main():
    print("=" * 60)
    print("AI-BASED 8-PUZZLE SOLVER")
    print("=" * 60)

    print("\nInitial state:")
    print_state(INITIAL_STATE)

    print("Goal state:")
    print_state(GOAL_STATE)

    if not is_solvable(INITIAL_STATE):
        print("The puzzle is NOT solvable.")
        return

    print("The puzzle is solvable.")

    print("\nInitial heuristic values:")
    print("Misplaced Tiles:",
          misplaced_tiles(INITIAL_STATE, GOAL_STATE))
    print("Manhattan Distance:",
          manhattan_distance(INITIAL_STATE, GOAL_STATE))

    results = []

    results.append(
        run_algorithm(
            "BFS",
            bfs,
            INITIAL_STATE,
            GOAL_STATE
        )
    )

    results.append(
        run_algorithm(
            "Hill Climbing + Misplaced Tiles",
            hill_climbing,
            INITIAL_STATE,
            GOAL_STATE,
            misplaced_tiles
        )
    )

    results.append(
        run_algorithm(
            "A* + Misplaced Tiles",
            a_star,
            INITIAL_STATE,
            GOAL_STATE,
            misplaced_tiles
        )
    )

    results.append(
        run_algorithm(
            "A* + Manhattan Distance",
            a_star,
            INITIAL_STATE,
            GOAL_STATE,
            manhattan_distance
        )
    )

    print("\n" + "=" * 85)
    print("PERFORMANCE COMPARISON")
    print("=" * 85)

    print(
        f"{'Algorithm':<35}"
        f"{'Moves':<10}"
        f"{'Nodes':<12}"
        f"{'Time(s)':<12}"
        f"{'Solved':<8}"
    )

    print("-" * 85)

    for result in results:
        path = result["path"]
        moves = len(path) - 1 if path else "-"
        solved = "Yes" if path else "No"

        print(
            f"{result['algorithm']:<35}"
            f"{str(moves):<10}"
            f"{result['nodes']:<12}"
            f"{result['time']:<12.6f}"
            f"{solved:<8}"
        )


if __name__ == "__main__":
    main()
```

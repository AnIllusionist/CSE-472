# Practical 2 — AI-Based Puzzle Solver Using Heuristic Search

## Objective

Develop an AI solver for an **8-Puzzle or Maze** using:

- **Hill Climbing**
- **A\* Search**

Compare their search efficiency and solution quality.

## 1. Choose One Problem

### Option A — 8-Puzzle

Example:

```text
1 2 3
4 5 6
7 _ 8
```

Goal:

```text
1 2 3
4 5 6
7 8 _
```

**State:** Complete board configuration.

### Option B — Maze

```text
S 0 0 1 0
1 0 0 1 0
0 0 0 0 0
0 1 1 1 0
0 0 0 0 G
```

`S` = Start, `G` = Goal, `1` = Wall, `0` = Free cell.

**State:** Current position.

## 2. Heuristic Function

A heuristic estimates how close a state is to the goal.

### 8-Puzzle
Use **Manhattan Distance**:

```text
h(n) = total horizontal + vertical distance
       of tiles from their goal positions
```

### Maze

```text
h(n) = Manhattan distance to the goal
```

## 3. Hill Climbing

```text
Current State
     ↓
Generate neighbors
     ↓
Choose a better neighbor
     ↓
Repeat
```

Stop when the goal is reached or no neighbor is better.

**Note:** Hill Climbing can get stuck at a local maximum/minimum, plateau, or dead end.

## 4. A* Search

Use:

\[
f(n)=g(n)+h(n)
\]

where:

- `g(n)` = cost from start to current state
- `h(n)` = estimated cost to the goal
- `f(n)` = estimated total cost

```text
Start
 ↓
Generate next states
 ↓
Calculate g(n), h(n), f(n)
 ↓
Choose best candidate
 ↓
Repeat until Goal
```

## 5. Minimum Requirements

Your program should:

1. Represent the problem state.
2. Generate valid neighboring states.
3. Implement Hill Climbing.
4. Implement A*.
5. Use the same test problem for both.
6. Display the final solution path if found.

## 6. Compare the Algorithms

| Measure | Hill Climbing | A* |
|---|---:|---:|
| Solution found? | | |
| Path cost | | |
| States explored | | |
| Execution time | | |

Answer briefly:

- Which algorithm explored fewer states?
- Which found a better solution?
- Did Hill Climbing get stuck?
- How did the heuristic help?
- Why can A* be more reliable?

## 7. Expected Output

```text
Initial State:
...

Goal State:
...

Hill Climbing:
Solution: Yes/No
Path Cost: ...
States Explored: ...
Time: ...

A*:
Solution: Yes/No
Path Cost: ...
States Explored: ...
Time: ...
```

## 8. Deliverable

Submit:

- Source code
- Output/screenshots
- Comparison table
- Short conclusion (4–5 lines)

### Conclusion

State which algorithm performed better for your chosen problem and how the heuristic affected search efficiency.

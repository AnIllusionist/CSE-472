# BFS & DFS Practical Class

## Artificial Intelligence — Uninformed Search

### Learning Objectives

By the end of this practical, students should be able to:

- Represent a problem's state space as a graph.
- Understand BFS and DFS as search strategies.
- Implement BFS using a **Queue**.
- Implement DFS using a **Stack**.
- Understand `append()`, `pop()`, `popleft()`, and `visited`.
- Trace both algorithms manually.
- Find a path from an initial state to a goal state.
- Visualize BFS and DFS.

---

# 1. Quick Recap: What Are We Searching?

A search problem contains:

| Component | Meaning |
|---|---|
| Initial State | Where we start |
| Actions | What we are allowed to do |
| Transitions | How an action changes the state |
| Goal State | What counts as success |
| State Space | All possible/reachable states |

Today's question is:

> **Once we have a state space, which state should we explore next?**

Two answers:

- **BFS — Breadth-First Search**
- **DFS — Depth-First Search**

---

# 2. Our State Space

We will use this graph throughout the practical:

```text
              A
            /   \\
           B     C
          / \\   / \\
         D   E F   G
             |
             H
```

Assume:

- `A` = Initial State
- `H` = Goal State
- Edges = possible actions/transitions

Our task:

> **Find H starting from A.**

---

# 3. Represent the Graph in Python

```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': ['H'],
    'F': [],
    'G': [],
    'H': []
}

print(graph)
```

Check the neighbours:

```python
print("Neighbours of A:", graph['A'])
print("Neighbours of B:", graph['B'])
print("Neighbours of E:", graph['E'])
```

---

# 4. The Key Difference Between BFS and DFS

The basic search process is almost identical.

Both algorithms:

1. Start from a node.
2. Remove a node from a data structure.
3. Check whether it was already visited.
4. Mark it as visited.
5. Check whether it is the goal.
6. Find its neighbours.
7. Add the neighbours.
8. Repeat.

The important difference is the data structure.

| | BFS | DFS |
|---|---|---|
| Data Structure | **Queue** | **Stack** |
| Rule | FIFO | LIFO |
| Remove | Front | Top |
| Add neighbours | Back | Top |
| Behaviour | Level by level | Deep first |

### Memory trick

```text
BFS → Queue → FIFO → Broad
DFS → Stack → LIFO → Deep
```

---

# 5. Stack Data Structure

DFS uses a **Stack**.

A stack follows:

> **LIFO — Last In, First Out**

Think of a stack of plates.

```text
       TOP
        ↓
      ┌───┐
      │ C │  ← Last added
      ├───┤
      │ B │
      ├───┤
      │ A │  ← First added
      └───┘
```

Python lists can act as stacks.

### Push / Add

```python
stack = []

stack.append('A')
stack.append('B')
stack.append('C')

print(stack)
```

### Pop / Remove

```python
x = stack.pop()

print("Removed:", x)
print("Stack:", stack)
```

The two operations are:

```python
stack.append(x)   # PUSH
stack.pop()       # POP
```

---

# 6. Queue Data Structure

BFS uses a **Queue**.

A queue follows:

> **FIFO — First In, First Out**

```text
FRONT                         BACK
  ↓                             ↓
┌───┬───┬───┐
│ A │ B │ C │
└───┴───┴───┘
  ↑
 removed first
```

Use `deque`:

```python
from collections import deque

queue = deque()

queue.append('A')
queue.append('B')
queue.append('C')

print(queue)
```

Remove from the front:

```python
x = queue.popleft()

print("Removed:", x)
print("Queue:", queue)
```

The two operations are:

```python
queue.append(x)    # Add to BACK
queue.popleft()    # Remove from FRONT
```

---

# 7. BFS Algorithm — Crisp

```text
1. Create Queue and Visited.
2. Append Start to Queue.
3. While Queue is not empty:
      a. Pop from FRONT.
      b. If already visited → skip.
      c. Add node to Visited.
      d. Check Goal.
      e. Find neighbours.
      f. Append neighbours to BACK of Queue.
4. Repeat.
```

In simple words:

> **Take the oldest node waiting in the queue, visit it, and put its neighbours at the back.**

---

# 8. DFS Algorithm — Crisp

```text
1. Create Stack and Visited.
2. Append Start to Stack.
3. While Stack is not empty:
      a. Pop from TOP.
      b. If already visited → skip.
      c. Add node to Visited.
      d. Check Goal.
      e. Find neighbours.
      f. Append neighbours to TOP of Stack.
4. Repeat.
```

In simple words:

> **Take the newest node in the stack, visit it, and put its neighbours on top.**

---

# 9. BFS and DFS Side by Side

| Step | BFS | DFS |
|---|---|---|
| 1 | Create **Queue + Visited** | Create **Stack + Visited** |
| 2 | `Queue.append(Start)` | `Stack.append(Start)` |
| 3 | Check Queue | Check Stack |
| 4 | `Queue.popleft()` | `Stack.pop()` |
| 5 | Check `Visited` | Check `Visited` |
| 6 | Add to `Visited` | Add to `Visited` |
| 7 | Check Goal | Check Goal |
| 8 | Find neighbours | Find neighbours |
| 9 | `Queue.append(neighbour)` | `Stack.append(neighbour)` |
| 10 | Repeat | Repeat |

### The main difference

```text
                 BFS                     DFS
                  ↓                       ↓
               QUEUE                    STACK
                  ↓                       ↓
                FIFO                    LIFO
                  ↓                       ↓
          Remove from FRONT        Remove from TOP
                  ↓                       ↓
          Explore LEVEL-WISE        Explore DEEPLY
```

---

# 10. BFS: Manual Trace

Graph:

```text
              A
            /   \\
           B     C
          / \\   / \\
         D   E F   G
             |
             H
```

Start:

```text
Queue = [A]
Visited = {}
```

### Step 1

Remove A:

```text
Queue = []
Visited = {A}
```

Add B and C:

```text
Queue = [B, C]
```

### Step 2

Remove B:

```text
Queue = [C]
Visited = {A, B}
```

Add D and E:

```text
Queue = [C, D, E]
```

### Step 3

Remove C:

```text
Queue = [D, E]
Visited = {A, B, C}
```

Add F and G:

```text
Queue = [D, E, F, G]
```

BFS explores **level by level**.

---

# 11. BFS Implementation

```python
from collections import deque

def bfs(graph, start):

    queue = deque()
    visited = set()

    queue.append(start)

    while queue:

        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        print(node, end=" ")

        for neighbour in graph[node]:

            if neighbour not in visited:
                queue.append(neighbour)
```

Run:

```python
bfs(graph, 'A')
```

---

# 12. BFS With Goal

```python
from collections import deque

def bfs_search(graph, start, goal):

    queue = deque([start])
    visited = set()

    while queue:

        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        print("Exploring:", node)

        if node == goal:
            print("Goal Found!")
            return True

        for neighbour in graph[node]:

            if neighbour not in visited:
                queue.append(neighbour)

    print("Goal not found.")
    return False
```

Run:

```python
bfs_search(graph, 'A', 'H')
```

---

# 13. BFS Path Finding

```python
from collections import deque

def bfs_path(graph, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:

        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbour in graph[node]:

            if neighbour not in visited:
                queue.append(path + [neighbour])

    return None
```

Run:

```python
path = bfs_path(graph, 'A', 'H')
print("BFS Path:", path)
```

Expected:

```text
BFS Path: ['A', 'B', 'E', 'H']
```

---

# 14. DFS: Manual Trace

Use the same graph.

Start:

```text
Stack = [A]
Visited = {}
```

### Step 1

Pop A:

```text
Stack = []
Visited = {A}
```

To explore B before C, push C first and then B:

```text
Stack = [C, B]
             ↑
            TOP
```

### Step 2

Pop B:

```text
Stack = [C]
Visited = {A, B}
```

B has D and E.

Push E first, then D:

```text
Stack = [C, E, D]
                 ↑
                TOP
```

### Step 3

Pop D:

```text
Stack = [C, E]
Visited = {A, B, D}
```

D has no neighbours.

### Step 4

Pop E:

```text
Stack = [C]
Visited = {A, B, D, E}
```

E has H:

```text
Stack = [C, H]
```

### Step 5

Pop H:

```text
H = Goal
```

DFS has reached the goal by going deep into the current branch.

---

# 15. DFS Implementation Using a Stack

```python
def dfs(graph, start):

    stack = []
    visited = set()

    stack.append(start)

    while stack:

        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        print(node, end=" ")

        for neighbour in reversed(graph[node]):

            if neighbour not in visited:
                stack.append(neighbour)
```

Run:

```python
dfs(graph, 'A')
```

---

# 16. DFS With Goal

```python
def dfs_search(graph, start, goal):

    stack = [start]
    visited = set()

    while stack:

        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        print("Exploring:", node)

        if node == goal:
            print("Goal Found!")
            return True

        for neighbour in reversed(graph[node]):

            if neighbour not in visited:
                stack.append(neighbour)

    print("Goal not found.")
    return False
```

Run:

```python
dfs_search(graph, 'A', 'H')
```

---

# 17. DFS Path Finding

```python
def dfs_path(graph, start, goal):

    stack = [[start]]
    visited = set()

    while stack:

        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbour in reversed(graph[node]):

            if neighbour not in visited:
                stack.append(path + [neighbour])

    return None
```

Run:

```python
path = dfs_path(graph, 'A', 'H')
print("DFS Path:", path)
```

---

# 18. Why Do We Need `Visited`?

Consider a cycle:

```text
A → B → C
    ↑   |
    └───┘
```

Without `visited`, the search could repeatedly do:

```text
B → C → B → C → B → C → ...
```

Therefore both algorithms use:

```python
visited = set()
```

and:

```python
if node in visited:
    continue
```

---

# 19. Why `reversed()` in DFS?

Suppose:

```python
graph['A'] = ['B', 'C']
```

If we push normally:

```python
stack.append('B')
stack.append('C')
```

we get:

```text
[C] ← TOP
[B]
```

So C is popped first.

If we want B first:

```python
for neighbour in reversed(graph[node]):
    stack.append(neighbour)
```

we push C first and B second:

```text
[B] ← TOP
[C]
```

So B is popped first.

> `reversed()` controls neighbour order. The essential DFS mechanism is still **Stack + LIFO**.

---

# 20. BFS vs DFS Example

Consider:

```text
             S
           /   \\
          A     B
          |     |
          C     G
          |
          D
          |
          E
          |
          G
```

There are two paths:

```text
S → B → G
```

and:

```text
S → A → C → D → E → G
```

BFS explores level by level and can find:

```text
S → B → G
```

DFS may choose A first and find:

```text
S → A → C → D → E → G
```

### Key lesson

> **DFS does not guarantee the shortest path.**

For an unweighted graph, BFS does guarantee the shortest path in terms of number of edges.

---

# 21. Debug BFS — See the Queue

```python
from collections import deque

def bfs_debug(graph, start):

    queue = deque([start])
    visited = set()

    while queue:

        print("\\n-------------------")
        print("Queue:", list(queue))

        node = queue.popleft()

        print("POP:", node)

        if node in visited:
            continue

        visited.add(node)

        print("Visited:", visited)

        for neighbour in graph[node]:

            if neighbour not in visited:
                queue.append(neighbour)

        print("Queue after adding neighbours:", list(queue))
```

Run:

```python
bfs_debug(graph, 'A')
```

Ask:

> **Which node will be removed next?**

---

# 22. Debug DFS — See the Stack

```python
def dfs_debug(graph, start):

    stack = [start]
    visited = set()

    while stack:

        print("\\n-------------------")
        print("Stack:", stack)

        node = stack.pop()

        print("POP:", node)

        if node in visited:
            continue

        visited.add(node)

        print("Visited:", visited)

        for neighbour in reversed(graph[node]):

            if neighbour not in visited:
                stack.append(neighbour)

        print("Stack after adding neighbours:", stack)
```

Run:

```python
dfs_debug(graph, 'A')
```

Ask:

> **Which node is now on TOP of the stack?**

---

# 23. Classroom Challenge — Predict Before Running

```python
challenge_graph = {
    'S': ['A', 'B', 'C'],
    'A': ['D', 'E'],
    'B': ['F'],
    'C': ['G', 'H'],
    'D': [],
    'E': ['I'],
    'F': [],
    'G': [],
    'H': [],
    'I': []
}
```

Before running, predict:

1. BFS traversal order.
2. DFS traversal order.
3. BFS path from `S` to `I`.
4. DFS path from `S` to `I`.

Then verify:

```python
print("BFS:")
bfs(challenge_graph, 'S')

print("\\nDFS:")
dfs(challenge_graph, 'S')

print("\\nBFS Path:")
print(bfs_path(challenge_graph, 'S', 'I'))

print("\\nDFS Path:")
print(dfs_path(challenge_graph, 'S', 'I'))
```

---

# 24. Practical Challenge — Build Your Own Graph

Create your own state space.

Possible themes:

- University campus
- Metro stations
- Maze
- Robot movement
- Game levels
- Web pages
- Treasure hunt

Requirements:

- At least 8 nodes.
- At least 2 branches.
- At least 1 dead end.
- At least 1 goal.
- Ideally include a cycle.

Then run:

```python
print("BFS:", bfs(my_graph, 'Start'))
print("DFS:", dfs(my_graph, 'Start'))
```

Answer:

1. What is the Initial State?
2. What are the Actions?
3. What are the Transitions?
4. What is the Goal State?
5. How does BFS explore your graph?
6. How does DFS explore your graph?

---

# 25. Visualization

Install:

```bash
pip install networkx matplotlib
```

Then:

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

for node, neighbours in graph.items():

    for neighbour in neighbours:
        G.add_edge(node, neighbour)

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 6))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1800,
    font_size=12,
    arrows=True
)

plt.title("AI State Space")
plt.show()
```

---

# 26. Visualize a Solution Path

```python
import networkx as nx
import matplotlib.pyplot as plt

def draw_solution(graph, path, title):

    G = nx.DiGraph()

    for node, neighbours in graph.items():

        for neighbour in neighbours:
            G.add_edge(node, neighbour)

    pos = nx.spring_layout(G, seed=42)

    path_edges = list(zip(path, path[1:]))

    edge_colors = [
        "red" if edge in path_edges else "gray"
        for edge in G.edges()
    ]

    node_colors = [
        "lightgreen" if node in path else "lightgray"
        for node in G.nodes()
    ]

    plt.figure(figsize=(8, 6))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=1800,
        width=2,
        font_size=12,
        arrows=True
    )

    plt.title(title)
    plt.show()
```

BFS:

```python
path = bfs_path(graph, 'A', 'H')

draw_solution(
    graph,
    path,
    "BFS Solution"
)
```

DFS:

```python
path = dfs_path(graph, 'A', 'H')

draw_solution(
    graph,
    path,
    "DFS Solution"
)
```

---

# 27. Optional: Step-by-Step Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_search(graph, start, algorithm="BFS"):

    G = nx.DiGraph()

    for node, neighbours in graph.items():

        G.add_node(node)

        for neighbour in neighbours:
            G.add_edge(node, neighbour)

    pos = nx.spring_layout(G, seed=42)

    visited = set()

    if algorithm == "BFS":
        from collections import deque
        frontier = deque([start])
    else:
        frontier = [start]

    while frontier:

        if algorithm == "BFS":
            node = frontier.popleft()
        else:
            node = frontier.pop()

        if node in visited:
            continue

        visited.add(node)

        neighbours = (
            graph[node]
            if algorithm == "BFS"
            else reversed(graph[node])
        )

        for neighbour in neighbours:

            if neighbour not in visited:
                frontier.append(neighbour)

        plt.figure(figsize=(8, 6))

        node_colors = [
            "lightgreen" if n in visited else "lightgray"
            for n in G.nodes()
        ]

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1800,
            font_size=12,
            arrows=True
        )

        plt.title(
            f"{algorithm} — Currently Exploring: {node}"
        )

        plt.show()

        input("Press Enter for next step...")
```

Run:

```python
visualize_search(graph, 'A', "BFS")
```

Then:

```python
visualize_search(graph, 'A', "DFS")
```

Ask:

> **What visual difference do you notice?**

---

# 28. Complexity

Let:

- `V` = number of vertices/states
- `E` = number of edges/transitions

For graph traversal with a visited set:

```text
BFS Time ≈ O(V + E)
DFS Time ≈ O(V + E)
```

However, their memory behaviour is different.

BFS can keep many frontier nodes in its queue.

DFS generally keeps a narrower active search frontier.

---

# 29. Common Mistakes

### Mistake 1 — Wrong DFS removal

Do not use:

```python
stack.pop(0)
```

Use:

```python
stack.pop()
```

### Mistake 2 — Wrong BFS removal

Do not use:

```python
queue.pop()
```

Use:

```python
queue.popleft()
```

### Mistake 3 — Forgetting `visited`

This can cause repeated exploration in cyclic graphs.

### Mistake 4 — Thinking DFS finds the shortest path

DFS does not guarantee the shortest path.

---

# 30. Quick Quiz

### Q1
Which data structure does BFS use?

A. Stack  
B. Queue  
C. Tree  
D. Set

### Q2
Which data structure does DFS use?

A. Stack  
B. Queue  
C. Heap  
D. Hash table

### Q3
What does FIFO mean?

### Q4
What does LIFO mean?

### Q5
Which operation removes the next element in BFS?

```python
queue.________()
```

### Q6
Which operation removes the next element in DFS?

```python
stack.________()
```

### Q7
Why do we maintain a `visited` set?

### Q8
Which algorithm explores level by level?

### Q9
Which algorithm explores deeply before backtracking?

### Q10
Does DFS guarantee the shortest path?

---

# 31. Answers

1. **B — Queue**
2. **A — Stack**
3. First In, First Out
4. Last In, First Out
5. `popleft()`
6. `pop()`
7. To avoid repeatedly exploring the same state.
8. BFS
9. DFS
10. No.

---

# 32. Exit Ticket

Answer before leaving:

### 1.

Complete:

```text
BFS → ______ → ______
DFS → ______ → ______
```

### 2.

What is the difference between:

```python
queue.popleft()
```

and:

```python
stack.pop()
```

### 3.

Why does DFS go deeper than BFS?

### 4.

For:

```text
        A
       / \\
      B   C
     / \\
    D   E
```

Write:

- BFS traversal
- DFS traversal

### 5.

Explain in one sentence:

> Why is BFS guaranteed to find the shortest path in an unweighted graph, while DFS is not?

---

# 33. Final Takeaway

```text
                    SEARCH
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
            BFS                 DFS
             ↓                   ↓
          QUEUE                STACK
             ↓                   ↓
           FIFO                 LIFO
             ↓                   ↓
       Level by Level        Deep First
             ↓                   ↓
       popleft()                pop()
```

### Remember

> **BFS and DFS follow almost the same search procedure. The major difference is the data structure used to decide which node is explored next.**

```text
BFS:
append → popleft
Queue → FIFO → Broad

DFS:
append → pop
Stack → LIFO → Deep
```

---

# 34. Looking Ahead

We have now explored a state space without using additional knowledge about which direction is promising.

Next question:

> **What if different paths have different costs?**

And then:

> **What if we have a clue about which state is closer to the goal?**

This leads naturally toward:

```text
BFS
 ↓
Cost-Based Search
 ↓
Heuristic Search
 ↓
A* Search
```

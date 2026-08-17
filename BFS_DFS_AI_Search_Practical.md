# Practical: Exploring State Spaces with BFS and DFS

> **Theme:** From "Which path do we explore first?" to working search
> algorithms.

## 1. Why are we doing this practical?

In the previous lecture, we learned that an AI problem can be formulated
using:

-   **Initial State** --- where we start
-   **Actions** --- what moves are allowed
-   **Transitions** --- what happens after an action
-   **Goal State** --- what counts as success
-   **Path Cost / Performance Measure** --- how we judge solutions

We also learned about the **state space**: all states reachable from the
initial state.

Today we answer the next question:

> **If there are many possible states, which one should the AI explore
> first?**

We will implement two classic **uninformed search** algorithms:

1.  **Breadth-First Search (BFS)**
2.  **Depth-First Search (DFS)**

------------------------------------------------------------------------

# 2. Today's Mission 🚑

Imagine an ambulance moving through connected intersections.

``` text
        A
       / \
      B   C
     / \   \
    D   E   F
         \ /
          G   ← Hospital
```

-   `A` = ambulance's starting location
-   `G` = hospital
-   Edges = roads the ambulance can use

Our task is to search the map and reach `G`.

Before coding, identify:

  Search Component   In Our Problem
  ------------------ ---------------------------------------------
  Initial State      A
  Actions            Move to a connected node
  Transition         Current node changes to the chosen neighbor
  Goal State         G
  State Space        A, B, C, D, E, F, G
  Solution           A path from A to G

------------------------------------------------------------------------

# 3. Representing a State Space in Python

We can represent the graph using an **adjacency list**.

``` python
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "G"],
    "F": ["C", "G"],
    "G": ["E", "F"]
}
```

Try:

``` python
print(graph["A"])
print(graph["B"])
```

Expected idea:

``` text
['B', 'C']
['A', 'D', 'E']
```

### Think before running

What does `graph["A"]` mean in terms of **actions**?

It tells us the states directly reachable from state `A`.

------------------------------------------------------------------------

# 4. Breadth-First Search (BFS)

## Intuition: Search level by level

Imagine searching for a friend in a college building.

You first check **all rooms on your current floor** before going to the
next floor.

BFS behaves similarly:

``` text
Start: A

Level 0:          A

Level 1:       B     C

Level 2:     D   E     F

Level 3:          G
```

BFS explores states that are **closest to the starting state first**.

## The Queue Idea

BFS uses a **Queue**.

A queue follows:

> **FIFO --- First In, First Out**

Like students standing in a canteen queue: the person who enters first
gets served first.

``` python
from collections import deque

queue = deque()

queue.append("A")
queue.append("B")
queue.append("C")

print(queue)

first = queue.popleft()
print("Removed:", first)
print("Remaining:", queue)
```

------------------------------------------------------------------------

# 5. Your First BFS

``` python
from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "G"],
    "F": ["C", "G"],
    "G": ["E", "F"]
}

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()

        if node not in visited:
            print(node, end=" ")
            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

bfs(graph, "A")
```

Expected traversal:

``` text
A B C D E F G
```

## Trace BFS manually

  Step   Removed   Queue after adding neighbors   Visited
  ------ --------- ------------------------------ ---------
  1      A         B, C                           A
  2      B         C, D, E                        A, B
  3      C         D, E, F                        A, B, C
  ...    ...       ...                            ...

### Mini Challenge

Complete the remaining rows yourself before moving ahead.

------------------------------------------------------------------------

# 6. BFS as Goal Search

Printing every node is useful for learning, but an AI search normally
asks:

> **Have I reached my goal?**

``` python
from collections import deque

def bfs_search(graph, start, goal):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()

        print("Exploring:", node)

        if node == goal:
            print("Goal found!")
            return True

        if node not in visited:
            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    print("Goal not found.")
    return False


bfs_search(graph, "A", "G")
```

### Question

Does BFS need to explore every state before finding `G`?

Not necessarily. It can stop as soon as the goal is found.

------------------------------------------------------------------------

# 7. Make BFS Return the Actual Path

An AI usually needs more than:

``` text
Goal found!
```

The ambulance needs the actual route.

We therefore store:

``` text
(current_node, path_taken_so_far)
```

``` python
from collections import deque

def bfs_path(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None


path = bfs_path(graph, "A", "G")
print("BFS Path:", path)
```

Possible result:

``` text
BFS Path: ['A', 'B', 'E', 'G']
```

------------------------------------------------------------------------

# 8. Depth-First Search (DFS)

Now imagine a different strategy.

Instead of checking every nearby option, you pick **one road and keep
going** until:

-   you reach the goal, or
-   you hit a dead end.

Then you **backtrack**.

That is DFS.

> **Go deep first. Backtrack when necessary.**

For our graph, one possible exploration is:

``` text
A → B → D
        ↓
     dead end

backtrack to B

A → B → E → G
```

------------------------------------------------------------------------

# 9. DFS Uses a Stack

A stack follows:

> **LIFO --- Last In, First Out**

Think of a stack of plates. The last plate placed on top is usually the
first one removed.

``` python
stack = []

stack.append("A")
stack.append("B")
stack.append("C")

print(stack)

last = stack.pop()

print("Removed:", last)
print("Remaining:", stack)
```

------------------------------------------------------------------------

# 10. Iterative DFS

``` python
def dfs(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node not in visited:
            print(node, end=" ")
            visited.add(node)

            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)


dfs(graph, "A")
```

A possible traversal is:

``` text
A B D E G F C
```

> The exact DFS order can change depending on the order in which
> neighbors are stored or pushed.

------------------------------------------------------------------------

# 11. Recursive DFS --- The Elegant Version

DFS can also be expressed naturally using recursion.

``` python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    visited.add(node)
    print(node, end=" ")

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)


dfs_recursive(graph, "A")
```

### What is recursion doing here?

Each function call effectively says:

> "Before I finish exploring this state, let me completely explore this
> unvisited neighbor."

The Python call stack therefore behaves like the explicit stack used in
iterative DFS.

------------------------------------------------------------------------

# 12. DFS Goal Search with a Path

``` python
def dfs_path(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None


path = dfs_path(graph, "A", "G")
print("DFS Path:", path)
```

------------------------------------------------------------------------

# 13. BFS vs DFS: Same Problem, Different Search Strategy

Run:

``` python
print("BFS:", bfs_path(graph, "A", "G"))
print("DFS:", dfs_path(graph, "A", "G"))
```

Then discuss:

  -----------------------------------------------------------------------
  BFS                                 DFS
  ----------------------------------- -----------------------------------
  Explores level by level             Explores one branch deeply

  Uses a queue                        Uses a stack / recursion

  Finds a shortest path in an         Does **not** guarantee the shortest
  **unweighted graph**                path

  Can require lots of memory          Often uses less memory

  Useful when the goal may be nearby  Useful when solutions may be deep
  -----------------------------------------------------------------------

### Important

"Shortest" here means **fewest edges/actions**, not necessarily shortest
travel time or distance.

If roads have different costs, BFS alone is not enough.

------------------------------------------------------------------------

# 14. 🔥 Experiment: Make BFS and DFS Behave Differently

Use this graph:

``` python
graph2 = {
    "S": ["A", "B"],
    "A": ["C"],
    "B": ["G"],
    "C": ["D"],
    "D": ["E"],
    "E": ["G"],
    "G": []
}

print("BFS:", bfs_path(graph2, "S", "G"))
print("DFS:", dfs_path(graph2, "S", "G"))
```

Before running, predict the paths.

``` text
S
├── A
│   └── C
│       └── D
│           └── E
│               └── G
└── B
    └── G
```

Ask yourself:

> Which algorithm is more likely to quickly notice the short `S → B → G`
> route?

------------------------------------------------------------------------

# 15. Visualize the Graph

For visualization, we can use `networkx` and `matplotlib`.

In Google Colab, these are usually available. If needed:

``` python
!pip install networkx matplotlib
```

Then:

``` python
import networkx as nx
import matplotlib.pyplot as plt

graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "G"],
    "F": ["C", "G"],
    "G": ["E", "F"]
}

G = nx.Graph()

for node in graph:
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1800,
    font_size=12
)

plt.title("Ambulance State Space")
plt.show()
```

------------------------------------------------------------------------

# 16. Visualize the BFS Solution Path

``` python
import networkx as nx
import matplotlib.pyplot as plt

path = bfs_path(graph, "A", "G")

G = nx.Graph()

for node in graph:
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

pos = nx.spring_layout(G, seed=42)

path_edges = list(zip(path, path[1:]))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1800
)

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=path_edges,
    width=4
)

plt.title("BFS Path: " + " → ".join(path))
plt.show()
```

------------------------------------------------------------------------

# 17. Visualize Search Step-by-Step

This version pauses after every explored node.

``` python
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bfs(graph, start, goal):
    G = nx.Graph()

    for node in graph:
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G, seed=42)

    queue = deque([start])
    visited = set()

    while queue:
        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        plt.clf()

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1800
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(visited),
            node_size=1800
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=2200
        )

        plt.title(
            f"Currently Exploring: {node}\n"
            f"Visited: {list(visited)}"
        )

        plt.pause(1)

        if node == goal:
            print("Goal found:", node)
            break

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)

    plt.show()


visualize_bfs(graph, "A", "G")
```

> Run this in a normal Python environment/Jupyter setup that supports
> interactive plotting. In some notebook environments, repeated frames
> may be displayed instead of a smooth animation.

------------------------------------------------------------------------

# 18. Visualize DFS Step-by-Step

``` python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_dfs(graph, start, goal):
    G = nx.Graph()

    for node in graph:
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G, seed=42)

    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        plt.clf()

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1800
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(visited),
            node_size=1800
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_size=2200
        )

        plt.title(
            f"Currently Exploring: {node}\n"
            f"Visited: {list(visited)}"
        )

        plt.pause(1)

        if node == goal:
            print("Goal found:", node)
            break

        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    plt.show()


visualize_dfs(graph, "A", "G")
```

------------------------------------------------------------------------

# 19. 🎮 Challenge: Campus Navigation

Suppose these locations are connected:

``` text
Hostel ─ Canteen ─ Library ─ Lab
   │         │
   │       Ground
   │         │
   └──── AcademicBlock ─ Auditorium
```

Represent it yourself:

``` python
campus = {
    "Hostel": [],
    "Canteen": [],
    "Library": [],
    "Lab": [],
    "Ground": [],
    "AcademicBlock": [],
    "Auditorium": []
}
```

### Your Tasks

1.  Complete the adjacency list.
2.  Use BFS to find a route from `Hostel` to `Lab`.
3.  Use DFS for the same problem.
4.  Print both paths.
5.  Visualize the campus graph.
6.  Change the neighbor order. Does DFS change?
7.  Does BFS still return a minimum-edge path?

------------------------------------------------------------------------

# 20. 🧩 Challenge: Can the Search Escape a Cycle?

Consider:

``` text
A → B → C
↑       ↓
└───────┘
```

What could happen if we do **not** maintain a `visited` set?

Try reasoning before coding.

The search may repeatedly revisit:

``` text
A → B → C → A → B → C → ...
```

This is why graph-search implementations keep track of visited states.

------------------------------------------------------------------------

# 21. Coding Exercise: Search Any Graph

Write one function:

``` python
def search(graph, start, goal, method):
    pass
```

It should work like:

``` python
print(search(graph, "A", "G", "bfs"))
print(search(graph, "A", "G", "dfs"))
```

Expected style of output:

``` text
['A', 'B', 'E', 'G']
['A', 'B', 'E', 'G']
```

The paths may differ depending on the graph and neighbor order.

### Hint

``` python
if method == "bfs":
    # use queue
elif method == "dfs":
    # use stack
```

------------------------------------------------------------------------

# 22. ⭐ Bonus: Let the Student Enter the Graph

``` python
graph = {}

n = int(input("How many nodes? "))

for _ in range(n):
    node = input("Node name: ")
    neighbors = input(
        f"Neighbors of {node} separated by spaces: "
    ).split()

    graph[node] = neighbors

start = input("Start node: ")
goal = input("Goal node: ")

print("Graph:", graph)
print("BFS Path:", bfs_path(graph, start, goal))
print("DFS Path:", dfs_path(graph, start, goal))
```

Example input:

``` text
How many nodes? 4

Node name: A
Neighbors of A separated by spaces: B C

Node name: B
Neighbors of B separated by spaces: D

Node name: C
Neighbors of C separated by spaces: D

Node name: D
Neighbors of D separated by spaces:

Start node: A
Goal node: D
```

------------------------------------------------------------------------

# 23. What Does This Have to Do with AI?

BFS and DFS are not just graph tricks.

They demonstrate the search idea introduced through **state spaces**.

A problem-solving agent can think in terms of:

``` text
Current State
     ↓
Possible Actions
     ↓
New States
     ↓
Search
     ↓
Goal State
```

Examples:

-   **Route planning:** states = locations
-   **8-Puzzle:** states = tile arrangements
-   **Chess:** states = board configurations
-   **Robot navigation:** states = robot positions/configurations
-   **Scheduling:** states = partial/complete assignments

The graph may be explicitly stored, or it may be generated as the AI
explores possible actions.

------------------------------------------------------------------------

# 24. Quick Concept Check

Answer without running code.

### Q1

Which data structure is naturally associated with BFS?

A. Stack\
B. Queue\
C. Dictionary only\
D. Recursion only

### Q2

Which strategy explores one branch deeply before backtracking?

A. BFS\
B. DFS

### Q3

In an unweighted graph, which algorithm guarantees a path with the
fewest edges?

A. BFS\
B. DFS

### Q4

Why do we maintain `visited`?

A. To make nodes look nicer\
B. To prevent unnecessary revisits and cycles\
C. To sort the graph\
D. To calculate road cost

### Q5

If two roads have very different travel times, does ordinary BFS
necessarily find the fastest route?

A. Yes\
B. No

------------------------------------------------------------------------

# 25. Exit Challenge 🚀

You are building an AI system to help a student find a study room.

Before coding, write:

``` text
Initial State:
Actions:
Transitions:
Goal State:
Performance Measure:
```

Then answer:

1.  What would a **state** represent?
2.  What would the **state space** contain?
3.  Could BFS be useful?
4.  When might BFS be a poor choice?
5.  How could road/room "costs" change the search problem?

------------------------------------------------------------------------

# 26. Practical Takeaway

By the end of this practical, you should be able to explain:

``` text
Problem Formulation
        ↓
State Space
        ↓
Search Strategy
     ↙       ↘
   BFS       DFS
 Queue      Stack
Level-wise  Depth-wise
```

The important idea is not memorizing code.

> **BFS and DFS are two different strategies for deciding which state in
> a state space should be explored next.**

That question --- *which path should we explore first?* --- is the
bridge from problem formulation to AI search.

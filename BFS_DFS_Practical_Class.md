# Practical Class: Exploring State Spaces with BFS and DFS

> **Theme:** *How does an AI decide which path to explore first?*\
> **Level:** Introductory AI practical\
> **Language:** Python\
> **Estimated time:** 90--120 minutes

------------------------------------------------------------------------

## 1. Why are we doing this practical?

In the previous lecture, we learned that an AI problem can be formulated
using:

-   **Initial State** --- where the agent starts
-   **Actions** --- what the agent is allowed to do
-   **Transitions** --- how an action changes the current state
-   **Goal State** --- what counts as success
-   **Path Cost / Performance Measure** --- how we judge one solution
    against another
-   **State Space** --- all states reachable by valid actions

Today we will make that idea **come alive in code**.

Instead of only looking at a state-space diagram, we will make a
computer **explore it** using two fundamental uninformed search
algorithms:

1.  **Breadth-First Search (BFS)** --- explore *level by level*
2.  **Depth-First Search (DFS)** --- explore *one path deeply before
    coming back*

### Today's challenge

Imagine that the graph below is a simplified campus map:

``` text
             A
           /   \
          B     C
         / \   / \
        D   E F   G
             |
             H
```

Suppose:

-   `A` = Main Gate
-   `B` = Academic Block
-   `C` = Hostel Area
-   `D` = Library
-   `E` = Cafeteria
-   `F` = Sports Complex
-   `G` = Parking
-   `H` = Hospital

**Question:** If an ambulance starts at `A` and needs to find `H`, which
locations should it explore first?

That depends on the **search strategy**.

------------------------------------------------------------------------

# 2. Setup

You can run this practical in:

-   Jupyter Notebook
-   Google Colab
-   VS Code
-   PyCharm
-   Any Python 3 environment

Check Python:

``` python
print("AI Search Practical")
print("Python is ready!")
```

For the visualization section, install:

``` bash
pip install networkx matplotlib
```

In Google Colab, these libraries are usually already available.

------------------------------------------------------------------------

# 3. Representing a State Space as a Graph

A graph contains:

-   **Nodes** → states
-   **Edges** → possible transitions/actions

We can represent our graph using a Python dictionary.

``` python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': [],
    'H': []
}

print(graph)
```

Here:

``` python
'A': ['B', 'C']
```

means:

> From state `A`, the agent can move to `B` or `C`.

Try:

``` python
print("Possible moves from A:", graph['A'])
print("Possible moves from C:", graph['C'])
print("Possible moves from H:", graph['H'])
```

### Think before running

What should each line print?

------------------------------------------------------------------------

# 4. First Exploration: Visit the Neighbours

Before BFS and DFS, let us understand how to access connected states.

``` python
current_state = 'A'

print("Current state:", current_state)

for neighbour in graph[current_state]:
    print("Can move to:", neighbour)
```

Expected idea:

``` text
Current state: A
Can move to: B
Can move to: C
```

This is the basic operation that search algorithms repeatedly perform.

------------------------------------------------------------------------

# 5. Breadth-First Search (BFS)

## 5.1 The intuition

Imagine searching for a friend in a university building.

You could:

1.  Check every room on the ground floor.
2.  Then every room on the first floor.
3.  Then every room on the second floor.
4.  Continue until you find the person.

That is the spirit of **Breadth-First Search**.

BFS explores:

> **Nearest states first, then moves outward level by level.**

For our graph:

``` text
             A          Level 0
           /   \
          B     C       Level 1
         / \   / \
        D   E F   G     Level 2
             |
             H          Level 3
```

A possible BFS exploration order is:

``` text
A → B → C → D → E → F → G → H
```

------------------------------------------------------------------------

# 6. Why BFS Uses a Queue

A **queue** follows:

> **FIFO --- First In, First Out**

Think of students standing in a canteen queue.

The student who entered first is served first.

Python provides an efficient queue using `deque`.

``` python
from collections import deque

queue = deque()

queue.append('A')
queue.append('B')
queue.append('C')

print(queue)

first = queue.popleft()

print("Removed:", first)
print("Queue now:", queue)
```

The first inserted item (`A`) is removed first.

------------------------------------------------------------------------

# 7. BFS Traversal --- Simple Version

``` python
from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': [],
    'H': []
}

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()

        if node not in visited:
            print(node, end=" ")

            visited.add(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

bfs(graph, 'A')
```

Expected output:

``` text
A B C D E F G H
```

------------------------------------------------------------------------

# 8. Understanding BFS Line by Line

The important structures are:

``` python
visited = set()
queue = deque([start])
```

`visited` remembers states already explored.

`queue` remembers states waiting to be explored.

Then:

``` python
while queue:
```

means:

> Continue searching while there is still something left to explore.

Next:

``` python
node = queue.popleft()
```

removes the **oldest** waiting node.

Then:

``` python
visited.add(node)
```

marks the state as explored.

Finally:

``` python
for neighbour in graph[node]:
    if neighbour not in visited:
        queue.append(neighbour)
```

adds possible next states to the queue.

------------------------------------------------------------------------

# 9. Watch BFS Think

This version prints the queue at every step.

``` python
from collections import deque

def bfs_debug(graph, start):
    visited = set()
    queue = deque([start])
    step = 1

    while queue:
        print(f"\n--- Step {step} ---")
        print("Queue before removal:", list(queue))

        node = queue.popleft()
        print("Exploring:", node)

        if node not in visited:
            visited.add(node)

            for neighbour in graph[node]:
                if neighbour not in visited and neighbour not in queue:
                    queue.append(neighbour)

        print("Visited:", list(visited))
        print("Queue after expansion:", list(queue))

        step += 1

bfs_debug(graph, 'A')
```

### Classroom activity

Before each step appears, pause and predict:

> **Which node will BFS remove next?**

------------------------------------------------------------------------

# 10. BFS That Actually Finds a Goal

Traversal is useful, but AI search normally wants to find a **goal**.

``` python
from collections import deque

def bfs_search(graph, start, goal):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()

        if node == goal:
            return True

        if node not in visited:
            visited.add(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

    return False

print(bfs_search(graph, 'A', 'H'))
```

Output:

``` text
True
```

But this only tells us that `H` exists.

An ambulance needs something more useful:

> **What path should I follow?**

------------------------------------------------------------------------

# 11. BFS That Returns the Path

Instead of storing only nodes in the queue, store complete paths.

``` python
from collections import deque

def bfs_path(graph, start, goal):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbour in graph[node]:
                new_path = path + [neighbour]
                queue.append(new_path)

    return None

path = bfs_path(graph, 'A', 'H')

print("BFS path:", path)
```

Expected result:

``` text
BFS path: ['A', 'C', 'F', 'H']
```

------------------------------------------------------------------------

# 12. BFS as an AI Search Problem

For the ambulance example:

  AI concept        Our program
  ----------------- ------------------------------
  Initial State     `A`
  State             Current graph node
  Actions           Move to a connected node
  Transition        Current node → neighbour
  Goal State        `H`
  State Space       All nodes reachable from `A`
  Search Strategy   BFS

Notice that BFS does not know that `H` is a hospital.

It only knows:

-   where it starts,
-   where it can move,
-   what state counts as the goal.

That is why **problem formulation comes before the algorithm**.

------------------------------------------------------------------------

# 13. Depth-First Search (DFS)

## 13.1 The intuition

Suppose you enter a maze.

Instead of checking all nearby corridors first, you choose one corridor
and keep walking.

You continue until:

-   you find the goal, or
-   you reach a dead end.

If you hit a dead end, you **backtrack** and try another path.

That is **Depth-First Search**.

DFS explores:

> **Go deep first. Backtrack when necessary.**

One possible DFS order for our graph is:

``` text
A → B → D → E → C → F → H → G
```

The exact order depends on how neighbours are stored.

------------------------------------------------------------------------

# 14. Why DFS Uses a Stack

A **stack** follows:

> **LIFO --- Last In, First Out**

Think of a stack of plates.

The last plate placed on top is usually the first one removed.

Python lists can be used as stacks.

``` python
stack = []

stack.append('A')
stack.append('B')
stack.append('C')

print(stack)

last = stack.pop()

print("Removed:", last)
print("Stack now:", stack)
```

`C` was inserted last, so it is removed first.

------------------------------------------------------------------------

# 15. DFS Traversal --- Iterative Version

``` python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': [],
    'H': []
}

def dfs(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node not in visited:
            print(node, end=" ")
            visited.add(node)

            # Reverse keeps left-to-right exploration intuitive.
            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)

dfs(graph, 'A')
```

Expected output:

``` text
A B D E C F H G
```

------------------------------------------------------------------------

# 16. Watch DFS Think

``` python
def dfs_debug(graph, start):
    visited = set()
    stack = [start]
    step = 1

    while stack:
        print(f"\n--- Step {step} ---")
        print("Stack before removal:", stack)

        node = stack.pop()
        print("Exploring:", node)

        if node not in visited:
            visited.add(node)

            for neighbour in reversed(graph[node]):
                if neighbour not in visited:
                    stack.append(neighbour)

        print("Visited:", list(visited))
        print("Stack after expansion:", stack)

        step += 1

dfs_debug(graph, 'A')
```

### Classroom prediction game

At each step ask:

> **What is on top of the stack?**

That node will be explored next.

------------------------------------------------------------------------

# 17. Recursive DFS

DFS can also be written naturally using recursion.

``` python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    visited.add(node)
    print(node, end=" ")

    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs_recursive(graph, neighbour, visited)

dfs_recursive(graph, 'A')
```

Expected output:

``` text
A B D E C F H G
```

### What recursion is doing

When DFS calls:

``` python
dfs_recursive(graph, neighbour, visited)
```

Python temporarily remembers the current function call.

This creates an implicit **call stack**.

So:

-   iterative DFS → we create the stack ourselves;
-   recursive DFS → Python manages the stack for us.

For learning search algorithms, understand the iterative version first
because the stack is visible.

------------------------------------------------------------------------

# 18. DFS That Finds a Path

``` python
def dfs_path(graph, start, goal):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbour in reversed(graph[node]):
                new_path = path + [neighbour]
                stack.append(new_path)

    return None

path = dfs_path(graph, 'A', 'H')

print("DFS path:", path)
```

For this graph, you should eventually get:

``` text
['A', 'C', 'F', 'H']
```

On a different graph, DFS may find a different---and possibly
longer---path than BFS.

------------------------------------------------------------------------

# 19. BFS vs DFS --- The Main Idea

``` text
BFS                              DFS

        A                               A
      /   \                           /
     B     C                         B
    / \   / \                       /
   D   E F   G                     D

Explore sideways first.           Explore deeply first.
```

  -----------------------------------------------------------------------
  Feature                 BFS                     DFS
  ----------------------- ----------------------- -----------------------
  Main data structure     Queue                   Stack

  Rule                    FIFO                    LIFO

  Exploration             Level by level          Deep path first

  Complete?               Yes, for finite         Not always in
                          branching               infinite-depth spaces

  Shortest path?          Yes for                 Not guaranteed
                          unweighted/equal-cost   
                          edges                   

  Memory                  Usually higher          Usually lower

  Good when               Goal may be near start  Memory is limited /
                                                  solutions may be deep
  -----------------------------------------------------------------------

### Important

**BFS does not automatically find the cheapest route when edges have
different costs.**

If one road takes 2 minutes and another takes 20 minutes, number of
edges alone is not enough. That leads to **cost-based search**, which
can be studied next.

------------------------------------------------------------------------

# 20. A Graph Where BFS and DFS Behave Differently

Let us make the difference more obvious.

``` python
graph2 = {
    'S': ['A', 'B'],
    'A': ['C'],
    'B': ['G'],
    'C': ['D'],
    'D': ['E'],
    'E': ['F'],
    'F': ['G'],
    'G': []
}
```

Visual idea:

``` text
S
├── A → C → D → E → F → G
└── B → G
```

There is a short path:

``` text
S → B → G
```

and a much longer path:

``` text
S → A → C → D → E → F → G
```

Run:

``` python
print("BFS:", bfs_path(graph2, 'S', 'G'))
print("DFS:", dfs_path(graph2, 'S', 'G'))
```

Possible result:

``` text
BFS: ['S', 'B', 'G']
DFS: ['S', 'A', 'C', 'D', 'E', 'F', 'G']
```

### Discussion

Why did BFS find the shorter path?

Because it explored all paths of depth 1 before depth 2, all depth-2
possibilities before depth 3, and so on.

Why did DFS take the long route?

Because it committed to the `A` branch and kept going deeper.

------------------------------------------------------------------------

# 21. Visualizing the Graph

Now let us draw the state space.

``` python
import networkx as nx
import matplotlib.pyplot as plt

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': [],
    'H': []
}

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

plt.title("Our AI State Space")
plt.show()
```

------------------------------------------------------------------------

# 22. Visualize BFS Step by Step

Run this cell and press **Enter** after each step.

``` python
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bfs(graph, start):
    G = nx.DiGraph()

    for node, neighbours in graph.items():
        G.add_node(node)

        for neighbour in neighbours:
            G.add_edge(node, neighbour)

    pos = nx.spring_layout(G, seed=42)

    visited = set()
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbour in graph[node]:
            if neighbour not in visited and neighbour not in queue:
                queue.append(neighbour)

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
            f"BFS exploring: {node}\n"
            f"Visited: {order}\n"
            f"Queue: {list(queue)}"
        )

        plt.show()

        input("Press Enter for the next BFS step...")

visualize_bfs(graph, 'A')
```

Green nodes are states BFS has already explored.

Watch how the green region expands **level by level**.

------------------------------------------------------------------------

# 23. Visualize DFS Step by Step

``` python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_dfs(graph, start):
    G = nx.DiGraph()

    for node, neighbours in graph.items():
        G.add_node(node)

        for neighbour in neighbours:
            G.add_edge(node, neighbour)

    pos = nx.spring_layout(G, seed=42)

    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbour in reversed(graph[node]):
            if neighbour not in visited:
                stack.append(neighbour)

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
            f"DFS exploring: {node}\n"
            f"Visited: {order}\n"
            f"Stack: {stack}"
        )

        plt.show()

        input("Press Enter for the next DFS step...")

visualize_dfs(graph, 'A')
```

Compare the visual pattern with BFS.

DFS should appear to form a **deep trail** before returning to
unexplored alternatives.

------------------------------------------------------------------------

# 24. Visualize the Final Search Path

This function highlights the solution found by a search algorithm.

``` python
import networkx as nx
import matplotlib.pyplot as plt

def draw_solution(graph, path, title="Solution Path"):
    G = nx.DiGraph()

    for node, neighbours in graph.items():
        G.add_node(node)

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

    plt.title(f"{title}: {' → '.join(path)}")
    plt.show()

bfs_solution = bfs_path(graph, 'A', 'H')
draw_solution(graph, bfs_solution, "BFS Solution")
```

------------------------------------------------------------------------

# 25. Search Until the Goal --- Animated Classroom Version

This version visualizes only until the goal is found.

``` python
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bfs_to_goal(graph, start, goal):
    G = nx.DiGraph()

    for node, neighbours in graph.items():
        G.add_node(node)
        for neighbour in neighbours:
            G.add_edge(node, neighbour)

    pos = nx.spring_layout(G, seed=42)

    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)

        plt.figure(figsize=(8, 6))

        node_colors = []
        for n in G.nodes():
            if n == node:
                node_colors.append("orange")
            elif n == goal:
                node_colors.append("lightblue")
            elif n in visited:
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightgray")

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
            f"Currently exploring: {node}\n"
            f"Current path: {' → '.join(path)}"
        )
        plt.show()

        if node == goal:
            print("GOAL FOUND!")
            print("Path:", " → ".join(path))
            return path

        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(path + [neighbour])

        input("Press Enter to continue searching...")

    print("Goal not found.")
    return None

visualize_bfs_to_goal(graph, 'A', 'H')
```

------------------------------------------------------------------------

# 26. A More Realistic Campus Search

Let us stop calling states `A`, `B`, and `C`.

``` python
campus = {
    'Main Gate': ['Block 1', 'Hostel'],
    'Block 1': ['Library', 'Cafeteria'],
    'Hostel': ['Sports Complex', 'Parking'],
    'Library': [],
    'Cafeteria': [],
    'Sports Complex': ['Hospital'],
    'Parking': [],
    'Hospital': []
}

start = 'Main Gate'
goal = 'Hospital'

print("BFS route:")
print(bfs_path(campus, start, goal))

print("\nDFS route:")
print(dfs_path(campus, start, goal))
```

Now the code starts looking like a tiny route-planning AI.

------------------------------------------------------------------------

# 27. What Happens If the Graph Has a Cycle?

Real state spaces are not always trees.

Consider:

``` text
A → B → C
↑       ↓
└───────┘
```

We can represent it as:

``` python
cyclic_graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A', 'D'],
    'D': []
}
```

What would happen if we kept following:

``` text
A → B → C → A → B → C → ...
```

forever?

That is why we maintain a:

``` python
visited
```

set.

Try:

``` python
print("BFS:")
bfs(cyclic_graph, 'A')

print("\nDFS:")
dfs(cyclic_graph, 'A')
```

Both terminate because already explored states are not repeatedly
expanded.

------------------------------------------------------------------------

# 28. Challenge 1 --- Predict Before Running

Do **not** run the code immediately.

``` python
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

Write down:

1.  BFS traversal order from `S`
2.  DFS traversal order from `S`
3.  BFS path from `S` to `I`
4.  DFS path from `S` to `I`

Then verify:

``` python
print("BFS traversal:")
bfs(challenge_graph, 'S')

print("\nDFS traversal:")
dfs(challenge_graph, 'S')

print("\nBFS path:", bfs_path(challenge_graph, 'S', 'I'))
print("DFS path:", dfs_path(challenge_graph, 'S', 'I'))
```

------------------------------------------------------------------------

# 29. Challenge 2 --- Build Your Own State Space

Create a graph with at least:

-   8 states
-   2 branches
-   1 goal
-   1 dead end
-   1 cycle

Example domains:

-   university campus
-   metro stations
-   rooms in a game
-   treasure hunt
-   robot moving between locations
-   web pages connected by links

Starter:

``` python
my_graph = {
    'Start': [],
    # Add your states here
}

start = 'Start'
goal = 'Goal'

print("BFS path:", bfs_path(my_graph, start, goal))
print("DFS path:", dfs_path(my_graph, start, goal))
```

### Your task

Answer:

-   What is the **initial state**?
-   What are the **actions**?
-   What are the **transitions**?
-   What is the **goal state**?
-   What would be a useful **performance measure**?

------------------------------------------------------------------------

# 30. Challenge 3 --- Who Explores Fewer States?

Modify BFS so it also counts how many nodes were explored.

``` python
from collections import deque

def bfs_with_count(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    explored_count = 0

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)
        explored_count += 1

        if node == goal:
            return path, explored_count

        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(path + [neighbour])

    return None, explored_count
```

Now do the same for DFS:

``` python
def dfs_with_count(graph, start, goal):
    stack = [[start]]
    visited = set()
    explored_count = 0

    while stack:
        path = stack.pop()
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)
        explored_count += 1

        if node == goal:
            return path, explored_count

        for neighbour in reversed(graph[node]):
            if neighbour not in visited:
                stack.append(path + [neighbour])

    return None, explored_count
```

Compare:

``` python
bfs_result = bfs_with_count(graph, 'A', 'H')
dfs_result = dfs_with_count(graph, 'A', 'H')

print("BFS:", bfs_result)
print("DFS:", dfs_result)
```

### Question

Is one algorithm **always** faster?

No. It depends on:

-   where the goal is,
-   how the state space branches,
-   the order of neighbours,
-   the depth of the solution.

------------------------------------------------------------------------

# 31. Challenge 4 --- Find a Missing Goal

What should happen if the goal does not exist?

``` python
print(bfs_path(graph, 'A', 'Z'))
print(dfs_path(graph, 'A', 'Z'))
```

Both should return:

``` text
None
```

An intelligent system must be able to report:

> **No solution was found.**

instead of crashing or searching forever.

------------------------------------------------------------------------

# 32. Mini Competition: Human vs AI Search

### Round 1

One student creates a graph on paper with 10--15 nodes.

Another student chooses a hidden goal.

The class predicts:

-   BFS order
-   DFS order
-   which algorithm will reach the goal first

Then enter the graph into Python and test it.

### Round 2

Move the goal to another node.

Ask:

> Did the better algorithm change?

### Lesson

There is no universally best search strategy for every problem.

The structure of the **state space** matters.

------------------------------------------------------------------------

# 33. Optional: Grid Search Like a Robot

A state does not have to be a letter.

It can be a coordinate:

``` python
start = (0, 0)
goal = (2, 2)
```

Here is a small grid:

``` text
S . .
. X .
. . G
```

`X` is blocked.

``` python
from collections import deque

grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]

start = (0, 0)
goal = (2, 2)

def get_neighbours(position, grid):
    row, col = position

    moves = [
        (-1, 0),   # Up
        (1, 0),    # Down
        (0, -1),   # Left
        (0, 1)     # Right
    ]

    neighbours = []

    for dr, dc in moves:
        new_row = row + dr
        new_col = col + dc

        if (
            0 <= new_row < len(grid)
            and 0 <= new_col < len(grid[0])
            and grid[new_row][new_col] == 0
        ):
            neighbours.append((new_row, new_col))

    return neighbours

print(get_neighbours(start, grid))
```

Now perform BFS:

``` python
def bfs_grid(grid, start, goal):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == goal:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbour in get_neighbours(current, grid):
            if neighbour not in visited:
                queue.append(path + [neighbour])

    return None

path = bfs_grid(grid, start, goal)

print("Robot path:", path)
```

Now we are very close to actual **robot navigation / route-planning
search**.

------------------------------------------------------------------------

# 34. Visualize the Grid Path

``` python
import matplotlib.pyplot as plt

def visualize_grid(grid, path=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap="Greys")

    if path:
        rows = [position[0] for position in path]
        cols = [position[1] for position in path]

        plt.plot(cols, rows, marker='o', linewidth=3)

    plt.xticks(range(len(grid[0])))
    plt.yticks(range(len(grid)))
    plt.grid(True)
    plt.title("Robot Search Path")
    plt.show()

visualize_grid(grid, path)
```

Try changing:

``` python
grid
start
goal
```

and observe how the solution changes.

------------------------------------------------------------------------

# 35. Complexity --- Just Enough for Today

Let:

-   `V` = number of vertices/states
-   `E` = number of edges/transitions

For graph traversal with a visited set:

``` text
BFS time  ≈ O(V + E)
DFS time  ≈ O(V + E)
```

But their **memory behaviour** and **order of exploration** differ.

In AI search, you will also often see complexity discussed using:

-   `b` = branching factor
-   `d` = depth of the shallowest solution
-   `m` = maximum depth

For now, remember the practical intuition:

> BFS may keep many frontier nodes in memory.\
> DFS usually stores a much narrower search path.

------------------------------------------------------------------------

# 36. Common Mistakes

## Mistake 1 --- Forgetting `visited`

Bad idea:

``` python
while queue:
    node = queue.popleft()

    for neighbour in graph[node]:
        queue.append(neighbour)
```

On cyclic graphs this may keep repeating states.

------------------------------------------------------------------------

## Mistake 2 --- Using `pop()` in BFS

``` python
node = queue.pop()
```

This removes from the end and starts behaving like a stack.

BFS needs:

``` python
node = queue.popleft()
```

------------------------------------------------------------------------

## Mistake 3 --- Using `popleft()` for DFS

DFS needs LIFO behaviour:

``` python
node = stack.pop()
```

------------------------------------------------------------------------

## Mistake 4 --- Assuming DFS finds the shortest path

It does not.

DFS finds **a** path, depending on exploration order.

------------------------------------------------------------------------

## Mistake 5 --- Thinking a node must be a location

A state can represent:

-   a board configuration,
-   a timetable,
-   a robot position,
-   a game situation,
-   a webpage,
-   a puzzle arrangement,
-   or any snapshot relevant to the problem.

------------------------------------------------------------------------

# 37. Quick Concept Check

Answer without looking above.

### Q1

Which data structure does BFS normally use?

``` text
A. Stack
B. Queue
C. Dictionary only
D. Array only
```

### Q2

Which principle describes a queue?

``` text
A. LIFO
B. FIFO
```

### Q3

Which search explores deeply before backtracking?

``` text
A. BFS
B. DFS
```

### Q4

For an unweighted graph, which algorithm guarantees a shortest path in
number of edges?

``` text
A. BFS
B. DFS
```

### Q5

Why do we need a `visited` set?

### Q6

Is a state always a physical location?

### Q7

If every road has a different travel time, is ordinary BFS enough to
guarantee the fastest route?

------------------------------------------------------------------------

# 38. Answers

1.  **B --- Queue**
2.  **B --- FIFO**
3.  **B --- DFS**
4.  **A --- BFS**
5.  To prevent repeated exploration and infinite loops in cyclic graphs.
6.  No. A state is a representation of the current situation of the
    problem.
7.  No. BFS minimizes the number of equal-cost steps, not arbitrary path
    cost.

------------------------------------------------------------------------

# 39. Exit Ticket

Before leaving, complete this without copying code.

Given:

``` python
graph = {
    'Home': ['Market', 'College'],
    'Market': ['Hospital'],
    'College': ['Library', 'Hospital'],
    'Library': [],
    'Hospital': []
}
```

Do the following:

1.  Write the BFS traversal from `Home`.
2.  Write the DFS traversal from `Home`.
3.  Find a path from `Home` to `Hospital` using BFS.
4.  Identify:
    -   Initial State
    -   Actions
    -   Transitions
    -   Goal State
5.  Explain in **one sentence** why BFS and DFS explore the same state
    space differently.
6.  Modify the graph by adding one cycle and confirm that your code
    still terminates.

------------------------------------------------------------------------

# 40. Complete Reference Code

Use this only **after** attempting the practical yourself.

``` python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(neighbour)

    return order


def dfs(graph, start):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbour in reversed(graph[node]):
            if neighbour not in visited:
                stack.append(neighbour)

    return order


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


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': [],
    'H': []
}

print("BFS traversal:", bfs(graph, 'A'))
print("DFS traversal:", dfs(graph, 'A'))

print("BFS path A → H:", bfs_path(graph, 'A', 'H'))
print("DFS path A → H:", dfs_path(graph, 'A', 'H'))
```

------------------------------------------------------------------------

# 41. What You Should Remember

If you remember only five things from today's practical, remember these:

1.  **AI search explores a state space.**
2.  **BFS uses a queue and explores level by level.**
3.  **DFS uses a stack and explores deeply before backtracking.**
4.  **A visited set prevents repeated exploration.**
5.  **The "best" search strategy depends on the problem and performance
    measure.**

------------------------------------------------------------------------

## One-line memory trick

``` text
BFS = Broad First = Queue = FIFO
DFS = Deep First  = Stack = LIFO
```

### Next step

Now that we can search a state space, the natural question is:

> **What if different paths have different costs, or we have information
> that tells us which direction looks more promising?**

That leads to **cost-based search, heuristics, and A\* search**.

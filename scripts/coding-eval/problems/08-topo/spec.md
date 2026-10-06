Write `topoSort(graph)`. `graph` is a plain object mapping each node name (string) to an array of the node
names it depends on. Nodes that appear only as dependencies are also nodes.

- Return an array of all nodes in which every node comes after all of its dependencies.
- When several nodes are ready at the same time, take the one that is smallest in string order first
  (this makes the result unique).
- If the graph has a cycle, throw an `Error` whose `cycle` property is an array of node names forming a
  cycle, with the first node repeated at the end, for example `["a", "b", "c", "a"]` where a depends on b,
  b on c and c on a. A self-dependency gives `["a", "a"]`.

## Setup

Run the `./setup.sh` script. It will download MP-SPDZ version 0.3.8 and run the unit tests for the BFS and maximal flow protocols.
   
## Execution

After executing the setup, you can run the protocols with the following commands.
These commands execute in the unrestricted network setting only.
Further, to execute these protocols in the semi-honest setting, specify `<sh/mal>` to be `sh`,
and `mal` to execute these protocols in the malicious-security setting.

- Logarithmic BFS with overflow mitigation: `./bfs.sh <sh/mal> logarithmic <number of nodes>`
- Logarithmic BFS without overflow mitigation: `./bfs.sh <sh/mal> logarithmic <number of nodes> with-overflows`
- Linear BFS: `./bfs.sh <sh/mal> linear <number of nodes>`
- BFS without comparisons: `./bfs.sh <sh/mal> no-comparison <number of nodes>`
- Optimized Blanton et al.: `./bfs.sh <sh/mal> blanton <number of nodes>`

- Edmonds-Carp: `./max-flow.sh <sh/mal> edmonds-karp <number of nodes>`
- Capacity-Scaling: `./max-flow.sh <sh/mal> capacity-scaling <number of nodes> <capacity-bit-length>`
- Dinic-Tarjan: `./max-flow.sh <sh/mal> dinic-tarjan <number of nodes>`
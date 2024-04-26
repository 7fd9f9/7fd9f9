import math
from dataclasses import dataclass
from Compiler.types import *
from Compiler.library import *

from .utils import permute_adjacency_matrix, argmin, clear_min, tree_select, print_list, find_first_nonzero_columnwise

BFS_TIMER_OFFSET = 10000
BFS_TIMER_FETCHING_NODE = BFS_TIMER_OFFSET + 1
BFS_TIMER_VISIT_NEIGHBORS = BFS_TIMER_OFFSET + 2
BFS_TIMER_UPDATE_STATE = BFS_TIMER_OFFSET + 3
BFS_TIMER_BUILDING_SELECTOR = BFS_TIMER_OFFSET + 4
BFS_TIMER_LOOP_BODY = BFS_TIMER_OFFSET + 5  # To debug, whole time of loop measured by the loop body
BFS_TIMER_LOOP = BFS_TIMER_OFFSET + 6  # To debug, whole time of loop measured from outside the loop
BFS_TIMER_INIT_STATE = BFS_TIMER_OFFSET + 7

BFS_LOGARITHMIC_OFFSET = BFS_TIMER_OFFSET + 10000
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY = BFS_LOGARITHMIC_OFFSET + 1
BFS_LOGARITHMIC_FIND_LAYERS = BFS_LOGARITHMIC_OFFSET + 2
BFS_LOGARITHMIC_FIND_SUBGRAPH = BFS_LOGARITHMIC_OFFSET + 3
BFS_LOGARITHMIC_FIND_TREE = BFS_LOGARITHMIC_OFFSET + 4
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_OVERFLOW_PREVENTION = BFS_LOGARITHMIC_OFFSET + 5
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE = BFS_LOGARITHMIC_OFFSET + 6
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_FINAL_CLEANUP = BFS_LOGARITHMIC_OFFSET + 7
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_MULTIPLY_OFFSET = BFS_LOGARITHMIC_OFFSET + 1000
BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_SQUARE_OFFSET = BFS_LOGARITHMIC_OFFSET + 2000

@dataclass
class BfsNodeState:
    is_visited: sint
    is_queued: sint
    is_unseen: sint
    layer_depth: sint

    @staticmethod
    def mux(cond: sint, if_true: "BfsNodeState", if_false: "BfsNodeState") -> "BfsNodeState":
        is_visited = cond.if_else(if_true.is_visited, if_false.is_visited)
        is_queued = cond.if_else(if_true.is_queued, if_false.is_queued)
        is_unseen = cond.if_else(if_true.is_unseen, if_false.is_unseen)
        layer_depth = cond.if_else(if_true.layer_depth, if_false.layer_depth)
        return BfsNodeState(is_visited, is_queued, is_unseen, layer_depth)


def bfs_logarithmic_layers_and_layered_subgraph(adj_matrix: Matrix, start_selector: Array,
                                                timers: bool = False, max_integer_value: int | None = None,
                                                compile_prints: bool = False) -> tuple[Matrix, Matrix]:
    """
    Finds the layers and the layered subgraph strating from the starting node in O(log n) communication rounds.
    Details in the paper.

    O(log n) rounds, O(n^3 log n) communications, O(n^3 log n) computation.

    :param adj_matrix: The adjacency matrix of the graph.
    :param start_selector: The selector array of the starting node.
    :param timers: Whether timers should be activated or not.
    :param max_integer_value: If not None, overflow mitigation will reduce intermediary values before the max_integer_value is exceeded.
    :param compile_prints: Additional compile-time debug prints.
    :return: Matrix containing selector arrays for layers 1 to n, and adjacency matrix of the layered subgraph.
    """

    n_nodes = adj_matrix.shape[0]

    if timers:
        start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY)

    if compile_prints:
        print("Compiling square and multiply.")

    nodes_after_steps = adj_matrix.same_shape()
    for i in range(n_nodes):
        nodes_after_steps[i][:] = start_selector[:]
    max_values = [1 for _ in range(n_nodes)]

    current_square_adj = adj_matrix.same_shape()
    current_square_adj[:] = adj_matrix[:]
    current_square_adj_max_value = 1

    for i in range(math.ceil(math.log2(n_nodes))):
        # Perform eventual reduction steps to prevent integer overflows.
        break_point()
        if timers:
            start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_OVERFLOW_PREVENTION)

        if max_integer_value is not None:
            if compile_prints:
                print(f"    Compiling value reductions in iteration {i + 1}")

            if n_nodes * current_square_adj_max_value ** 2 > max_integer_value:
                print(f"Reducing square matrix to prevent integer overflows at power {i}.")
                # for x in range(n_nodes):
                #     for y in range(n_nodes):
                #         current_square_adj[x][y] = current_square_adj[x][y] != 0
                current_square_adj[:] = current_square_adj[:] != 0
                current_square_adj_max_value = 1
                break_point()

            for j in range(n_nodes):
                if (j >> i) & 1 == 1:
                    if (max_integer_value is not None and
                            n_nodes * max_values[j] * current_square_adj_max_value > max_integer_value):
                        print(f"Reducing after_steps {j} to prevent integer overflows in round {i}.")
                        # for k in range(n_nodes):
                        #     nodes_after_steps[j][k] = (nodes_after_steps[j][k] != 0)
                        nodes_after_steps[j][:] = nodes_after_steps[j][:] != 0
                        max_values[j] = 1

            if compile_prints:
                print(f"        Done")

        # Square and multiply.
        if timers:
            stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_OVERFLOW_PREVENTION)
            start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE)
            start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_MULTIPLY_OFFSET + i)
            if compile_prints:
                print(f"Start timer {BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_MULTIPLY_OFFSET + i}")

        if compile_prints:
            print(f"     Compiling value updates / matrix multiplications in iteration {i + 1}.")

        break_point()
        # For whatever reason, this does not parallelize, leading to effective O(n) communication rounds.
        # for j in range(n_nodes):
        #     if (j >> i) & 1 == 1:
        #         nodes_after_steps[j][:] = current_square_adj.direct_mul(nodes_after_steps[j])
        #         max_values[j] *= n_nodes * current_square_adj_max_value

        # Alternative approach: Multiply everything, pick the interesting results.
        packing = []
        for j in range(n_nodes):
            if (j >> i) & 1 == 1:
                packing.append(j)
        packed_nodes_after_steps = sint.Matrix(rows=len(packing), columns=n_nodes)
        for k, j in enumerate(packing):
            packed_nodes_after_steps[k][:] = nodes_after_steps[j][:]
        packed_nodes_after_steps[:] = packed_nodes_after_steps.direct_mul(current_square_adj)
        for k, j in enumerate(packing):
            nodes_after_steps[j][:] = packed_nodes_after_steps[k][:]

        if timers:
            stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_MULTIPLY_OFFSET + i)
            start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_SQUARE_OFFSET + i)
            if compile_prints:
                print(f"Start timer {BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_SQUARE_OFFSET + i}")

        current_square_adj[:] = current_square_adj.direct_mul(current_square_adj)
        current_square_adj_max_value = n_nodes * current_square_adj_max_value ** 2

        if timers:
            stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE_SQUARE_OFFSET + i)
            stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_UPDATE)

    break_point()
    if timers:
        start_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_FINAL_CLEANUP)

    if compile_prints:
        print(f"     Square and multiply final reduction.")

    # nodes_after_steps = Matrix.create_from([[x > 0 for x in row] for row in nodes_after_steps])
    nodes_after_steps[:] = nodes_after_steps[:] != 0
    break_point()

    if compile_prints:
        print(f"        Done")

    if timers:
        stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY_FINAL_CLEANUP)
        stop_timer(BFS_LOGARITHMIC_SQUARE_AND_MULTIPLY)
        start_timer(BFS_LOGARITHMIC_FIND_LAYERS)

    if compile_prints:
        print(f"Compiling: Find layers.")

    layers = find_first_nonzero_columnwise(nodes_after_steps, compile_prints)

    break_point()

    if compile_prints:
        print(f"    Done")
        print(f"Compiling: Finding layered subgraph.")

    if timers:
        stop_timer(BFS_LOGARITHMIC_FIND_LAYERS)
        start_timer(BFS_LOGARITHMIC_FIND_SUBGRAPH)

    # Old approach: Produces A LOT of instructions => requires a lot of RAM during compilation.
    # layered_adj_matrix = Matrix.create_from([
    #     [adj_matrix[i][j] * sint.dot_product([layers[k][i] for k in range(0, n_nodes - 1)],
    #                                          [layers[k][j] for k in range(1, n_nodes)])
    #      for j in range(n_nodes)
    #      ]
    #     for i in range(n_nodes)
    # ])

    layered_adj_matrix_helper_matrix1 = sint.Matrix(rows=n_nodes, columns=n_nodes - 1)
    layered_adj_matrix_helper_matrix2 = sint.Matrix(rows=n_nodes - 1, columns=n_nodes)

    @for_range(n_nodes)
    def create_helper_matrix_outer(i: cint) -> None:
        @for_range(n_nodes - 1)
        def create_heler_matrix_inner(k: cint) -> None:
            layered_adj_matrix_helper_matrix1[i][k] = layers[k][i]
            layered_adj_matrix_helper_matrix2[k][i] = layers[k + 1][i]

    layered_adj_matrix = adj_matrix.same_shape()
    layered_adj_matrix[:] = adj_matrix[:] * layered_adj_matrix_helper_matrix1.direct_mul(
        layered_adj_matrix_helper_matrix2)

    break_point()

    if compile_prints:
        print(f"    Done")

    if timers:
        stop_timer(BFS_LOGARITHMIC_FIND_SUBGRAPH)

    return layers, layered_adj_matrix


def bfs_logarithmic(adj_matrix: Matrix, start_selector: Array, timers: bool = False,
                    max_integer_value: int | None = 2 ** 63,
                    compile_prints: bool = False) -> tuple[list[BfsNodeState], Matrix]:
    """
    The logarithmic BFS as described in the paper, but returns the transposed tree matrix for more seamless
    augmenting path extraction.

    Complexities: O(log n) rounds, O(n^3 log n) communication, O(n^3 log n) computation.
    """
    n_nodes = adj_matrix.shape[0]

    # First, find the layered subgraph.
    # This is outsourced to a seperate method as the Dinic-Tarjan max-flow protocol requires the layered
    # subgraph only.
    # O(log n) rounds, O(n^3 log n) communication, O(n^3 log n) computation.
    layers, layered_adj_matrix = bfs_logarithmic_layers_and_layered_subgraph(adj_matrix, start_selector, timers, max_integer_value, compile_prints)

    break_point()

    if compile_prints:
        print(f"Compiling: Finding BFS tree.")

    if timers:
        start_timer(BFS_LOGARITHMIC_FIND_TREE)

    # Filter the layered subgraph to the tree matrix.
    # O(1) rounds, O(n^2) communication, O(n^2) computation.
    tree_matrix = find_first_nonzero_columnwise(layered_adj_matrix, compile_prints)

    if compile_prints:
        print(f"    Done")
        print(f"Compiling: Rest.")

    # Return results.
    # 0 round, 0 communication, O(n) computation.
    seen = start_selector.same_shape()
    seen[:] = start_selector[:]

    depths = start_selector.same_shape()
    depths[:] = 0

    for i in range(1, n_nodes):
        seen[:] += layers[i][:]
        depths[:] += i * layers[i][:]

    if timers:
        stop_timer(BFS_LOGARITHMIC_FIND_TREE)

    node_states = [
        BfsNodeState(is_visited=seen[i],
                     is_queued=sint(0),
                     is_unseen=1 - seen[i],
                     layer_depth=depths[i]
                     ) for i in range(n_nodes)
    ]

    if compile_prints:
        print(f"    Done")

    return node_states, tree_matrix.transpose()


def bfs_naive(adj_matrix: Matrix, start_selector: Array, timers: bool = False) -> tuple[list[BfsNodeState], Matrix]:
    if timers:
        start_timer(BFS_TIMER_INIT_STATE)

    def cmp_states(a: BfsNodeState, b: BfsNodeState) -> sint:
        return (b.layer_depth + n_nodes * (1 - b.is_queued)) < (a.layer_depth + n_nodes * (1 - a.is_queued))

    n_nodes = adj_matrix.shape[0]
    node_states = [
        BfsNodeState(sint(0),
                     start_selector[i],
                     1 - start_selector[i],
                     sint(0))
        for i in range(n_nodes)
    ]

    parent_selectors = Matrix.create_from([
        [sint(0) for j in range(n_nodes)] for i in range(n_nodes)
    ])

    # print_ln("INIT:")
    # print_bfs_state(node_states)
    # print_ln("\n\n")

    current_selector = Array.create_from(start_selector)
    if timers:
        stop_timer(BFS_TIMER_INIT_STATE)
        start_timer(BFS_TIMER_LOOP)

    @for_range(n_nodes)
    def f(i):
        if timers:
            start_timer(BFS_TIMER_LOOP_BODY)

        # print_ln(f"BEFORE ITER %s:", i)
        # print_bfs_state(node_states)
        # print_ln("\n\n")
        if timers:
            start_timer(BFS_TIMER_FETCHING_NODE)
        adj_row = Array.create_from([
            sint.dot_product(adj_matrix.transpose()[j], current_selector) for j in range(n_nodes)
        ])
        current_layer_depth = sint.dot_product(current_selector,
                                               Array.create_from([state.layer_depth for state in node_states]))
        current_node_is_enqueued = sint.dot_product(current_selector,
                                                    Array.create_from([state.is_queued for state in node_states]))
        if timers:
            stop_timer(BFS_TIMER_FETCHING_NODE)
        # print_ln(f"ITER %s:", i)
        # print_list(current_selector, "Selector")
        # print_list(adj_row, "Neighbors")
        # print_ln("Current depth: %s", current_layer_depth.reveal())
        # print_ln("Current node is enqueued: %s", current_node_is_enqueued.reveal())

        # Enqueue all unseen neighboring nodes.
        if timers:
            start_timer(BFS_TIMER_VISIT_NEIGHBORS)
        for j, state in enumerate(node_states):
            newly_enqueued_state = BfsNodeState(cint(0), cint(1), cint(0), current_layer_depth + 1)
            update_node = current_node_is_enqueued * adj_row[j] * state.is_unseen
            new_state = BfsNodeState.mux(update_node, newly_enqueued_state, state)

            node_states[j].is_visited.update(new_state.is_visited)
            node_states[j].is_queued.update(new_state.is_queued)
            node_states[j].is_unseen.update(new_state.is_unseen)
            node_states[j].layer_depth.update(new_state.layer_depth)

            old_parent_selector = parent_selectors[j]
            new_parent_selector = old_parent_selector + update_node * current_selector[:]
            parent_selectors.assign_part_vector(new_parent_selector, j)
        if timers:
            stop_timer(BFS_TIMER_VISIT_NEIGHBORS)

        # Set the current node to be visited
        if timers:
            start_timer(BFS_TIMER_UPDATE_STATE)
        for state, sel in zip(node_states, current_selector):
            state.is_visited.update(state.is_visited + sel * current_node_is_enqueued)
            state.is_queued.update(state.is_queued - sel * current_node_is_enqueued)
        if timers:
            stop_timer(BFS_TIMER_UPDATE_STATE)

        # print_ln(f"AFTER ITER %s:", i)
        # print_bfs_state(node_states)
        # print_ln("Parent matrix")
        # for row in parent_selectors:
        #     print_list(row)
        # print_ln("\n\n")

        # Find the next node to be visited
        if timers:
            start_timer(BFS_TIMER_BUILDING_SELECTOR)
        current_selector.assign(tree_select(node_states, cmp_states, BfsNodeState.mux))
        if timers:
            stop_timer(BFS_TIMER_BUILDING_SELECTOR)
        if timers:
            stop_timer(BFS_TIMER_LOOP_BODY)

    if timers:
        stop_timer(BFS_TIMER_LOOP)
    return node_states, parent_selectors


def bfs_optimized_blanton_et_al(adj_matrix: Matrix, start_selector: Array, timers: bool = False) -> tuple[list[BfsNodeState], Matrix]:
    """
    Performs a breadth first search. This variation permutes the adjacency matrix and reveals the order in which the
    permuted nodes are visited.

    O(n log n) rounds, O(n^2) communication, O(n^2) computation.
    :param adj_matrix: The adjacency matrix of the graph.
    :param start_selector: The selector indicating the starting node.
    :param timers: Whether extra timers (used for debugging purposes) should be activated.
    :return: (node states, parent matrix)
            The node states indicate whether a node was seen or not and contain the layer depth.
            The parent_matrix[i][j] = 1 iff node j is the parent of node i in the BFS tree.
    """

    if timers:
        start_timer(BFS_TIMER_INIT_STATE)

    n_nodes = adj_matrix.shape[0]
    permutation = sint.get_secure_shuffle(n_nodes)

    # Permute the adjacency matrix.
    shuffled_matrix = permute_adjacency_matrix(adj_matrix, permutation)
    # Find the permuted index of the start node.
    current_node: cint = sum(
        [a * b for (a, b) in zip(range(n_nodes), start_selector[:].secure_permute(permutation))]).reveal()

    visit_order = Array.create_from([sint(i) for i in range(n_nodes)])
    # It is fine to re-use the same permutation.
    # As a consequence of this, "plain trace" of this algorithm is equivalent to the deterministic naïve BFS implementation.
    # Hence, the plain trace of the algorithm does not depend on the permutation and this is secure :)
    visit_order.secure_permute(permutation)

    # Initialize state.
    visited = start_selector.same_shape()
    plain_visited = cint.Array(n_nodes)

    enqueued = start_selector.same_shape()
    enqueued[current_node] = 1

    layer_depths = start_selector.same_shape()

    parent_selectors = adj_matrix.same_shape()
    parent_selectors.assign_all(0)

    if timers:
        stop_timer(BFS_TIMER_INIT_STATE)
        start_timer(BFS_TIMER_LOOP)

    # Perform the BFS.
    # O(n log n) rounds, O(n^2) comm, O(n^2) comp.
    @for_range(n_nodes)
    def f(i):
        # print_ln("Iteration %s", i)
        # print_ln("Visiting node %s", current_node)
        # print_list(visited, "Visited")
        # print_list(enqueued, "Enqueued")
        # print_list(layer_depths, "Layer depths")

        if timers:
            start_timer(BFS_TIMER_LOOP_BODY)
            start_timer(BFS_TIMER_FETCHING_NODE)

        # Fetch information about the current node.
        # 0 rounds, 0 comm, O(1) comp.
        current_layer_depth = layer_depths[current_node]
        adj_row = shuffled_matrix[current_node]
        current_node_is_enqueued = enqueued[current_node]

        if timers:
            stop_timer(BFS_TIMER_FETCHING_NODE)
            start_timer(BFS_TIMER_VISIT_NEIGHBORS)

        # Visit all neighbors in parallel.
        # 4 rounds, O(n) comm, O(n) comp.
        for j in range(n_nodes):
            update_node = current_node_is_enqueued * adj_row[j] * (1 - plain_visited[j] - enqueued[j])

            layer_depths[j] = update_node.if_else(current_layer_depth + 1, layer_depths[j])
            enqueued[j] += update_node
            parent_selectors[j][current_node] = update_node

        if timers:
            stop_timer(BFS_TIMER_VISIT_NEIGHBORS)
            start_timer(BFS_TIMER_UPDATE_STATE)

        # Update the visited node's state.
        # 0 round, 0 comm, O(1) comp.
        enqueued[current_node] -= current_node_is_enqueued
        visited[current_node] += current_node_is_enqueued
        plain_visited[current_node] = 1

        if timers:
            stop_timer(BFS_TIMER_UPDATE_STATE)

        # Find the next node to be visited.
        # O(log n) rounds, O(n) comm, O(n) comp.
        @if_(i + 1 < n_nodes)
        def determine_next_node():
            # TODO: Use plain_visited to limit the search space a-prior at the expense of larger compilation times
            if timers:
                start_timer(BFS_TIMER_BUILDING_SELECTOR)

            tmp_layer_depths = layer_depths.same_shape()
            # The "+ n_nodes * plain_visited" is needed to ensure that no node is visited twice, which would cause
            # a data leakage.
            tmp_layer_depths[:] = n_nodes * (
                    layer_depths[:] + n_nodes * (1 - enqueued[:]) + n_nodes * plain_visited) + visit_order[:]
            current_node.update(argmin(tmp_layer_depths).reveal())

            if timers:
                stop_timer(BFS_TIMER_BUILDING_SELECTOR)

        if timers:
            stop_timer(BFS_TIMER_LOOP_BODY)

    # Unshuffle the states.
    parent_selectors = permute_adjacency_matrix(parent_selectors, permutation, reverse=True)
    visited.secure_permute(permutation, reverse=True)
    layer_depths.secure_permute(permutation, reverse=True)

    node_states = [
        BfsNodeState(
            is_visited=visited[i],
            is_queued=sint(0),
            is_unseen=1 - visited[i],
            layer_depth=layer_depths[i]
        ) for i in range(n_nodes)
    ]
    return node_states, parent_selectors


class SelectorQueue:
    selector_size: int
    queue_max_size: int
    queue_state: Matrix
    pos: int

    def __init__(self, selector_size: int, queue_max_size: int, force_all_appear: bool = False):
        self.selector_size = selector_size
        self.queue_max_size = queue_max_size
        self.queue_state = sint.Matrix(rows=queue_max_size, columns=selector_size)
        self.queue_state.assign_all(sint(0))

        if force_all_appear:
            for i in range(selector_size):
                self.queue_state[-1][i] = 1

        self.pos = 0

    def add(self, nodes: Array) -> None:
        self.queue_state.assign_part_vector(nodes.get_vector(), self.pos)
        self.pos += 1

    def pop(self) -> Array:
        selector = self.peek()
        self.remove(selector)
        return selector

    def peek(self) -> Array:
        def op(left, right):
            return (1 - left)  # Choose left if it is 1, 0 otherwise

        big_selector = tree_select(self.queue_state.get_vector(), op)
        return Array.create_from(
            [sum([big_selector[j * self.selector_size + i] for j in range(self.queue_max_size)]) for i in
             range(self.selector_size)])

    def remove(self, selector: Array):
        selector_vec = Array.create_from(list(1 - selector.get_vector()) * self.queue_max_size).get_vector()
        self.queue_state.assign_vector(self.queue_state.get_vector() * selector_vec)


def bfs_comparison_free(adj_matrix: Matrix, start_selector: Array, timers: bool = False) -> tuple[list[BfsNodeState], Matrix]:
    """
    Performs a breadth first search. This variation avoids comparisons by having a bigger state, which was experimentally shown to
    be faster.

    O(n log n) comm rounds, O(n^3) comm, O(n^3) comp.
    :param adj_matrix: The adjacency matrix of the graph.
    :param start_selector: The selector indicating the starting node.
    :param timers: Whether extra timers (used for debugging purposes) should be activated.
    :return: (node states, parent matrix)
            The node states indicate whether a node was seen or not and contain the layer depth.
            The parent_matrix[i][j] = 1 iff node j is the parent of node i in the BFS tree.
    """
    if timers:
        start_timer(BFS_TIMER_INIT_STATE)

    # Init states.
    # 0 rounds, 0 comm, O(n) comp.
    n_nodes = adj_matrix.shape[0]
    node_states = [
        BfsNodeState(sint(0),
                     start_selector[i],
                     1 - start_selector[i],
                     sint(0))
        for i in range(n_nodes)
    ]

    parent_selectors = Matrix.create_from([
        [sint(0) for j in range(n_nodes)] for i in range(n_nodes)
    ])

    # print_ln("INIT:")
    # print_bfs_state(node_states)
    # print_ln("\n\n")

    # Create a selector queue and add the starting node.
    # 0 rounds, 0 comm, O(n^2) comp.
    queue = SelectorQueue(n_nodes, n_nodes)
    queue.add(start_selector)

    if timers:
        stop_timer(BFS_TIMER_INIT_STATE)
        start_timer(BFS_TIMER_LOOP)

    # O(n log n) rounds, O(n^3) comm, O(n^3) comp.
    @for_range(n_nodes)
    def f(i):
        if timers:
            start_timer(BFS_TIMER_LOOP_BODY)
        # print_ln(f"BEFORE ITER %s:", i)
        # print_bfs_state(node_states)
        # print_ln("\n\n")

        # Fetch the next node to be visited from the queue.
        # If the queue is empty, this will result in a "null node" (all entries set to zero).
        # O(log n) rounds, O(n^2) comm, O(n^2) comp.
        if timers:
            start_timer(BFS_TIMER_BUILDING_SELECTOR)
        current_selector = queue.pop()
        if timers:
            stop_timer(BFS_TIMER_BUILDING_SELECTOR)

        # Fetch the neighboring node selectors and the current layer depth.
        # 1 round, O(n) / O(n^2) comm depending on scalar product, O(n^2) comp.
        if timers:
            start_timer(BFS_TIMER_FETCHING_NODE)
        adj_row = Array.create_from([
            sint.dot_product(adj_matrix.transpose()[j], current_selector) for j in range(n_nodes)
        ])
        current_layer_depth = sint.dot_product(current_selector,
                                               Array.create_from([state.layer_depth for state in node_states]))
        if timers:
            stop_timer(BFS_TIMER_FETCHING_NODE)

        # print_ln(f"ITER %s:", i)
        # print_list(current_selector, "Selector")
        # print_list(adj_row, "Neighbors")
        # print_ln("Current depth: %s", current_layer_depth.reveal())
        # print_ln("Current node is enqueued: %s", current_node_is_enqueued.reveal())

        # Visit all neighboring nodes and update their states.
        # 3 rounds, O(n^2) comm, O(n^2) comp.
        if timers:
            start_timer(BFS_TIMER_VISIT_NEIGHBORS)
        nodes_to_enqueue = []
        for j, state in enumerate(node_states):
            # Determine if node j should be enqueued.
            # 1 round, O(1) comm, O(1) comp.
            update_node = adj_row[j] * state.is_unseen
            nodes_to_enqueue.append(update_node)

            # Update node j's state.
            # 1 round, O(1) comm, O(1) comp.
            newly_enqueued_state = BfsNodeState(cint(0), cint(1), cint(0), current_layer_depth + 1)
            new_state = BfsNodeState.mux(update_node, newly_enqueued_state, state)
            node_states[j].is_visited.update(new_state.is_visited)
            node_states[j].is_queued.update(new_state.is_queued)
            node_states[j].is_unseen.update(new_state.is_unseen)
            node_states[j].layer_depth.update(new_state.layer_depth)

            # Simultaneously to updating node j's state:
            # Update node j's parent selector.
            # 1 round (but count 0 because it is merged into the previous round),
            # O(n) comm, O(n) comp.
            old_parent_selector = parent_selectors[j]
            new_parent_selector = old_parent_selector + update_node * current_selector[:]
            parent_selectors.assign_part_vector(new_parent_selector, j)

        # Add newly visited nodes to queue.
        # 0 rounds, 0 comm, O(n) comp.
        queue.add(Array.create_from(nodes_to_enqueue))
        if timers:
            stop_timer(BFS_TIMER_VISIT_NEIGHBORS)

        # Set the current node to be visited.
        # 0 rounds, 0 comm, O(n) comp.
        if timers:
            start_timer(BFS_TIMER_UPDATE_STATE)
        for state, sel in zip(node_states, current_selector):
            state.is_visited.update(state.is_visited + sel)
            state.is_queued.update(state.is_queued - sel)
        if timers:
            stop_timer(BFS_TIMER_UPDATE_STATE)

        # print_ln(f"AFTER ITER %s:", i)
        # print_bfs_state(node_states)
        # print_ln("Parent matrix")
        # for row in parent_selectors:
        #     print_list(row)
        # print_ln("\n\n")

        if timers:
            stop_timer(BFS_TIMER_LOOP_BODY)

    if timers:
        stop_timer(BFS_TIMER_LOOP)
    return node_states, parent_selectors


def bfs_linear(adj_matrix: Matrix, start_selector: Array, timers: bool = False) -> tuple[list[BfsNodeState], Matrix]:
    """
    Performs a breadth-first search. This variation visits all nodes in one layer simultaneously, allowing a
    communication round complexity of O(n).
    This is achieved by performing O(n^2) comparisons in parallel.

    Complexities: O(n) rounds, O(n^3) communication, O(n^3) computation.
    :param adj_matrix: The adjacency matrix of the graph.
    :param start_selector: The selector indicating the starting node.
    :param timers: Whether extra timers (used for debugging purposes) should be activated.
    :return: (node states, parent matrix)
            The node states indicate whether a node was seen or not and contain the layer depth.
            The parent_matrix[i][j] = 1 iff node j is the parent of node i in the BFS tree.
    """

    # Initialize state.
    # 0 rounds, 0 comm, O(n) comp.
    n_nodes = adj_matrix.shape[0]

    current_layer = start_selector.same_shape()
    current_layer[:] = start_selector[:]

    layer_depths = start_selector.same_shape()
    layer_depths.assign_all(0)

    visited = start_selector.same_shape()
    visited.assign_all(0)

    parent_matrix = adj_matrix.same_shape()
    parent_matrix.assign_all(0)

    # Perform the search.
    # O(n) rounds, O(n^3) comm, O(n^3) comp.
    @for_range(n_nodes - 1)
    def iteration(current_iteration: cint) -> None:
        visited[:] += current_layer[:]

        # Find all connections from the current layer into the next layer.
        # 2 rounds, O(n^2) comm, O(n^2) comp.
        current_layer_edges = Matrix.create_from([
            [current_layer[i] * adj_matrix[i][j] * (1 - visited[j]) for j in range(n_nodes)] for i in range(n_nodes)
        ])

        # Now, the issue is that multiple nodes from the current layer might point towards the one node in the next layer.
        # To fix this, we filter the edges such that only node with the lowest id has an edge towards a node with multiple incoming edges.
        # This is achieved by first calculating the prefix-sums over each columns, and then only keeping edges where
        # the prefix-sum is zero.

        # Calculate the prefix-sum of each column.
        # 0 rounds, 0 comm, O(n^2) comp.
        prefix_sum = sint.Matrix(rows=n_nodes - 1, columns=n_nodes)
        prefix_sum[0][:] = current_layer_edges[0][:]
        for i in range(1, n_nodes - 1):
            prefix_sum[i][:] = prefix_sum[i - 1][:] + current_layer_edges[i][:]

        # Determine which edges are actually used.
        # O(1) rounds, O(n^2) comm, O(n^2) comp.
        filtered_layer_edges = Matrix.create_from([
            [current_layer_edges[i][j] if i == 0 else current_layer_edges[i][j] * (prefix_sum[i - 1][j] == 0) for j in
             range(n_nodes)]
            for i in range(n_nodes)
        ])

        # Another nice property of the filtered edges are that each column noch contains at most one 1, hence
        # we can determine the next layer by calculating the sums of the columns.
        # 0 rounds, 0 comm, O(n^2) comp.
        next_layer = Array.create_from(
            [sum([filtered_layer_edges[i][j] for i in range(n_nodes)]) for j in range(n_nodes)])

        # Update state.
        # 0 rounds, 0 comm, O(n^2) comp.
        layer_depths[:] += (current_iteration + 1) * next_layer[:]
        parent_matrix[:] += filtered_layer_edges.transpose()[:]
        current_layer[:] = next_layer[:]

    # Build the resulting node states.
    # 0 rounds, 0 comm, O(n) comp.
    node_states = [
        BfsNodeState(
            is_visited=visited[i],
            is_queued=sint(0),
            is_unseen=1 - visited[i],
            layer_depth=layer_depths[i]
        ) for i in range(n_nodes)
    ]
    return node_states, parent_matrix


bfs = bfs_logarithmic

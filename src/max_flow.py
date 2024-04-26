import math

from Compiler.library import *
from Compiler.types import *

from .bfs import bfs, bfs_logarithmic_layers_and_layered_subgraph
from .utils import clear_min, print_list, \
    determine_path_matrix


def max_flow_edmonds_karp(cap_matrix: Matrix, source_selector: Array, sink_selector: Array,
                          max_flow_value: int | cint | None = None,
                          max_capacity: int | cint | None = None,
                          max_iterations: int | cint| None = None,
                          starting_flow: Matrix | None = None,
                          bfs_algorithm=bfs,
                          infinity: int = 2 ** 40,
                          logarithmic_path_construction: bool = True) -> Matrix:
    """
    Calculates a maximum flow according to the Edmonds-Karp algorithm, i.e., it finds augmenting paths using a
    breadth-first-search.

    Usually, this requires O(n^3) breadth-first-searches, leading to O(n^3 log n) rounds, O(n^6 log n) communication, and O(n^6 log n) computation.
    However, if the value of the maximal flow f is known a-priori, for example because the graph has a known structure
    that allows inferring an upper bound of the maximal flow, the number of iterations can be reduced to f.
    This works because in each iteration, either at least one unit of flow is sent from the source to the sink, or an
    maximal flow has already been reached.
    An easy way to infer an upper bound of the maximal flow is provided by the max_capacity parameter that translates
    to a maximal flow value of n*max_capacity (f <= maximal outflow of the source <= n*max_capacity).

    In the case a bound f < n^3 is known, the algorithm has a complexity of
     O(f log n) rounds, O(f n^3 log n) communication, and O(f n^3 log n) computation.
    :param cap_matrix: The capacity matrix.
    :param source_selector: The selector array of the source node.
    :param sink_selector: The selector array of the sink node.
    :param starting_flow: An optional starting point for the algorithm. Useful if a flow just needs to be extended, and more tight iteration bounds can be applied.
    :param max_flow_value: An optional upper bound on the maximal flow value of the graph.
    :param max_capacity: An optional upper bound on the capacity of edges.
    :param bfs_algorithm: The underlying breath-first-search algorithm.
    :param infinity: A really large number that is functionally equivalent to infinity. Must be larger than all edge capacities.
    :param logarithmic_path_construction: Whether the square-and-multiply approach should be used to find an augmenting path based on the BFS's results. Default: True
    :return: A matrix indicating the maximal flow from the source to the sink on the graph.
    """
    flow = cap_matrix.same_shape()
    if starting_flow:
        flow[:] = starting_flow[:]
    else:
        flow.assign_all(0)

    n_nodes = cap_matrix.shape[0]

    iteration_bounds = [n_nodes ** 3]
    if max_flow_value is not None:
        iteration_bounds.append(max_flow_value)
    if max_capacity is not None:
        iteration_bounds.append(n_nodes * max_capacity)
    if max_iterations is not None:
        iteration_bounds.append(max_iterations)
    n_iterations = clear_min(iteration_bounds)

    @for_range(n_iterations)
    def _(iter):
        # Calculate the residual capacities.
        # 0 rounds, 0 communication, O(n^2) computation.
        residuals = cap_matrix.same_shape()
        residuals.assign_vector(cap_matrix.get_vector() - flow.get_vector())

        # Find the edges in the residual graph.
        # O(1) rounds, O(n^2) communication, O(n^2) computation.
        # Afterwards, apply breadth-first-search.
        # O(n log n) rounds, O(n^3) communication, O(n^3) computation.
        node_states, parents = bfs_algorithm(Matrix.create_from([
            [sint(x > 0) for x in row]
            for row in residuals
        ]), source_selector)

        # Reconstruct the shortest path from the source to the sink.
        if logarithmic_path_construction:
            # O(log n) rounds, O(n^3 log n) communication, O(n^3 log n) computation.
            path_matrix = determine_path_matrix(parents, sink_selector)
        else:
            # O(n) rounds, O(n^3) communication, O(n^3) computation.
            path_matrix = cap_matrix.same_shape()
            path_matrix.assign_all(sint(0))

            current_selector = sink_selector
            for path_iter in range(n_nodes):
                next_selector = Array.create_from([
                    sint.dot_product(current_selector, parents.get_column(i)) for i in range(n_nodes)
                ])
                path_matrix_diff = Matrix.create_from([
                    [current_selector[j] * next_selector[i] for j in range(n_nodes)] for i in range(n_nodes)
                ])
                path_matrix.iadd(path_matrix_diff)

                current_selector = next_selector

        # print_ln("Path %s", iter)
        # for row in path_matrix:
        #     print_list(row)

        # Find the smallest capacity along the path.
        # O(log n) rounds, O(n^2) communication, O(n^2) computation.
        search_matrix = infinity * (1 - path_matrix[:]) + residuals[:]
        augmented_capacity = util.min(search_matrix)

        # Augment the flow.
        # 1 round, O(n^2) communication, O(n^2) computation.
        flow[:] += augmented_capacity * (path_matrix[:] - path_matrix.transpose()[:])

    return flow


def max_flow_capacity_scaling(cap_matrix: Matrix, source_selector: Array, sink_selector: Array, max_capacity: int,
                              leak_better_step_iterations: bool = False,
                              infinity: int = 2**40, bfs_algorithm=bfs, logarithmic_path_construction: bool = True) -> Matrix:
    """
    Capacity scaling as described in the paper. Reduces the maximal flow problem to a series of "easy" max flow
    problems that can be solved efficiently using the Edmonds-Karp algorithm.

    :param cap_matrix: The capacity matrix of the graph.
    :param source_selector: Source selector indicating the source node.
    :param sink_selector: Source selector indicating the sink node.
    :param max_capacity: An upper bound on the edge capacities.
    :param infinity: A very large number, it must be bigger than the sum of all edge capacities.
    :param bfs_algorithm: The BFS to use.
    :param logarithmic_path_construction: Whether the augmenting path should be extracted in O(log n) round instead of O(n).
    :returns: The maximal flow matrix.
    """

    n_nodes = cap_matrix.shape[0]
    bit_length = math.ceil(math.log2(max_capacity))

    bit_decompositions = [[x.bit_decompose(bit_length) for x in row] for row in cap_matrix]
    reduced_capacity_matrices = []
    iterations = []
    for capacity_length in range(1, bit_length + 1):
        step_matrix = cap_matrix.same_shape()
        step_iterations = 0

        for i in range(n_nodes):
            for j in range(n_nodes):
                step_matrix[i][j] = sum([bit_decompositions[i][j][-k - 1] * 2 ** (capacity_length - k - 1) for k in range(capacity_length)])
                step_iterations += bit_decompositions[i][j][-capacity_length]

        reduced_capacity_matrices.append(step_matrix)
        iterations.append(step_iterations)

    if leak_better_step_iterations:
        iterations = [x.reveal() for x in iterations]
        print_ln("Iterations: " + "%s " * len(iterations), *iterations)

    flow = cap_matrix.same_shape()
    flow.assign_all(0)

    for (step_capacity_matrix, step_iterations) in zip(reduced_capacity_matrices, iterations):
        if not leak_better_step_iterations:
            # Upper bound on step iterations:
            step_iterations = n_nodes * (n_nodes - 1)

        flow[:] *= 2
        # O(n^2 log n) or O(step_iterations * log n) rounds
        flow = max_flow_edmonds_karp(step_capacity_matrix, source_selector, sink_selector,
                                     starting_flow=flow,
                                     max_iterations=step_iterations,
                                     infinity=infinity,
                                     bfs_algorithm=bfs_algorithm,
                                     logarithmic_path_construction=logarithmic_path_construction)
    return flow


def blocking_flow_tarjan(cap_matrix: Matrix, source_selector: Array, sink_selector: Array,
                         layer_selectors: Matrix) -> Matrix:
    """
    Finds a blocking flow in an acyclic graph.
    A blocking a flow is a flow such that there is a saturated edge on every path from the source to the sink [1].
    Note that is not necessarily a maximal flow, there might still be augmenting paths, but those would require
    re-routing some flow [1].

    The algorithm needs to iterate over the nodes in topological order, which has to be provided by the user as the
    layer_selectors parameters. A specialty of this implementation is that it can operate on multiple nodes at once,
    as long as those nodes are in the same layer of the BFS tree spanned from the source node.
    Refer to "max_flow_dinic_tarjan" for a data-oblivious way to create these layers.

    FOR THE CORRECTNESS OF THIS IMPLEMENTATION, IT IS IMPORTANT THAT THE SINK IS THE ONLY NODE ON THE SINK LAYER.
    In other words, you need to ensure that the last layer selector contains only the sink, and that every node
    that is reachable from the source is contained in at least one layer selector.
    max_flow_dinic_tarjan handles this by cutting of any nodes on the same layer as the sink, it is suggested
    to refer to that implementation.

    Complexities: O(n^2) rounds, O(n^4) communication, O(n^4) computation.

    [1]: Tarjan, “A Simple Version of Karzanov’s Blocking Flow Algorithm.”
    :param cap_matrix: The capacity matrix of the graph.
    :param source_selector: The selector array for the source node.
    :param sink_selector: The selector array for the sink node.
    :param layer_selectors: An array of "selector arrays" for each layer of the BFS tree spanned from the source node.
        These selector arrays contain ones at the position of the nodes in the corresponding layers and zeroes otherwise.
    :return:
    """
    n_nodes = cap_matrix.shape[0]

    # Remove the layer containing the source and the sink.
    # 1 round, O(n^2) comm, O(n^2) comp.
    actual_layer_selectors = sint.Matrix(rows=layer_selectors.shape[0] - 2, columns=n_nodes)
    for i in range(1, layer_selectors.shape[0] - 1):
        layer = layer_selectors[i]
        layer_has_sink = sint.dot_product(layer, sink_selector)
        actual_layer_selectors[i - 1][:] = (1 - layer_has_sink) * layer[:]
    n_layers = actual_layer_selectors.shape[0]

    # Initialize blocked array and flow / reverse flow / excess array.
    # 0 rounds, 0 comm, O(n) comp.
    blocked = source_selector.same_shape()
    blocked[:] = source_selector[:]

    flow = cap_matrix.same_shape()
    reverse_flow = cap_matrix.same_shape()
    excess = sint.Array(n_nodes)

    # Initialize pre-flow, i.e. the source sends as much flow as possible to the first-layer nodes.
    # 1 round, O(n^2) comm, O(n^2) comp.
    for i in range(n_nodes):
        i_is_source = source_selector[i]
        for j in range(n_nodes):
            edge_flow = cap_matrix[i][j] * i_is_source
            flow[i][j] = edge_flow
            reverse_flow[j][i] = -edge_flow
            excess[j] += edge_flow

    def print_state():
        print_ln("Flow:")
        for row in flow:
            print_list(row)
        print_ln("Excess:")
        print_list(excess)
        print_ln("Blocked:")
        print_list(blocked)

    # O(n^2) rounds, O(n^4) comm, O(n^4) comp.
    @for_range(n_nodes)
    def main_iteration(outer_iter):
        # print_ln("Iter %s-a", outer_iter)
        # print_state()
        # print_ln("")

        # Perform the increment flow step.
        # O(n) rounds, O(n^3) comm, O(n^3) comp.
        @for_range(n_layers)
        def increment_flow_iter(increment_flow_iter):
            layer_selector = actual_layer_selectors[increment_flow_iter]

            flow_deltas = flow.same_shape()
            flow_deltas.assign_all(0)
            excess_deltas = cap_matrix.same_shape()
            excess_deltas.assign_all(0)
            blocked_deltas = blocked.same_shape()
            blocked_deltas.assign_all(0)

            # Run increase flow on all nodes in the current layer in parallel.
            # O(1) rounds, O(n^2) comm, O(n^2) comp.
            for i in range(n_nodes):
                # 2 or 3 rounds (1 in ATLAS), O(n) comm, O(n) comp.
                possible_outflow = Array.create_from(
                    [layer_selector[i] * (1 - blocked[i]) * (cap_matrix[i][j] - flow[i][j]) * (1 - blocked[j]) for j in
                     range(n_nodes)])

                # 0 rounds, 0 comm, O(n) comp.
                previous_outflow = 0
                outflows = []
                for x in possible_outflow:
                    outflows.append(excess[i] - previous_outflow)
                    previous_outflow += x

                # Clamp the outflows,
                # O(1) rounds, O(n) comm, O(n) comp.
                outflows = [
                    f - ((f < 0) * f) - ((f > possible) * (f - possible))
                    for f, possible in zip(outflows, possible_outflow)
                ]

                # Update flows and excesses,
                # 0 rounds, 0 comm, O(n) comp.
                for j, f in enumerate(outflows):
                    flow_deltas[i][j] += f
                    excess_deltas[i][j] += f
                excess_deltas[i][i] -= sum(outflows)

                # Update block, i.e. activate block if
                # 1. the node is not already blocked, and
                # 2. there is still excess flow, and
                # 3. the node is not the sink. (the sink must remain unblocked)
                # O(1) rounds, O(1) comm, O(1) comp.
                blocked_deltas[i] += (1 - blocked[i]) * ((excess[i] + excess_deltas[i][i]) > 0) * (1 - sink_selector[i])

            # Update the variables.
            blocked[:] += blocked_deltas[:]
            flow.iadd(flow_deltas)
            reverse_flow[:] = reverse_flow[:] - flow_deltas.transpose()[:]
            for i in range(n_nodes):
                excess[:] += excess_deltas[i][:]

        # print_ln("Iter %s-b", outer_iter)
        # print_state()
        # print_ln("")

        # Perform the decrement flow step.
        # O(n) rounds, O(n^3) comm, O(n^3) comp.
        @for_range(n_layers)
        def decrement_flow_iter(decrement_flow_iter):
            layer_selector = actual_layer_selectors[n_layers - decrement_flow_iter - 1]

            flow_deltas = flow.same_shape()
            flow_deltas.assign_all(0)
            excess_deltas = cap_matrix.same_shape()
            excess_deltas.assign_all(0)

            # Perform all decrements in parallel.
            # O(1) rounds, O(n^2) comm, O(n^2) comp.
            for i in range(n_nodes):
                # 1 round, O(n) comm, O(n) comp.
                possible_outflow = [layer_selector[i] * blocked[i] * flow[j][i] for j in range(n_nodes)]

                # 0 rounds, 0 comm, O(n) comp.
                previous_outflow = 0
                outflows = []
                for x in possible_outflow:
                    outflows.append(excess[i] - previous_outflow)
                    previous_outflow += x

                # Clamp the outflows,
                # O(1) rounds, O(n) comm, O(n) comp.
                outflows = [
                    f - ((f < 0) * f) - ((f > possible) * (f - possible))
                    for f, possible in zip(outflows, possible_outflow)
                ]

                # Update flows and excesses,
                # 0 rounds, 0 comm, O(n) comp.
                for j, f in enumerate(outflows):
                    flow_deltas[j][i] -= f
                    excess_deltas[i][j] += f
                excess_deltas[i][i] -= sum(outflows)

            flow.iadd(flow_deltas)
            reverse_flow[:] = reverse_flow[:] - flow_deltas.transpose()[:]
            for i in range(n_nodes):
                excess[:] += excess_deltas[i][:]

    # print_state()
    flow.iadd(reverse_flow)
    return flow


def max_flow_dinic_tarjan(cap_matrix: Matrix, source_selector: Array, sink_selector: Array,
                          max_flow_value: int | None = None,
                          use_bfs_logarithmic_layered_graph: bool = True,
                          bfs_algorithm=bfs,
                          timers: bool = False) -> Matrix:
    """
    Calculates a maximal flow in the graph usinc Dinic's algorithm.
    To describe this algorithm on a very high level, we compare to Edmonds-Karps algorithm, which always augments along
    a shortest path from s to t, which guarantees that the max flow is found after O(nm)=O(n^3) augmentations.
    Dinic's algorithm is somewhat similar, but augments along "all" shortest paths from s to t in one iteration.
    This guarantees that the algorithm finishes within n iterations.

    More precisely, in each algorithm builds the layered subgraph of the residual graph, i.e. the subgraph that only
    contains edges that are part of a shortest path from the source to the sink.
    Then, it finds a blocking flow in the layered graph. This implementation uses Tarjan's blocking flow algorithm
    to find the blocking flow in O(n^2) rounds, resulting in a round complexity of O(n^3).
    The communication complexity is O(n^5) and the computational complexity is O(n^5).

    As with Edmonds-Karp algorithm, each iteration either increases the flow or a maximal flow was already achieved
    in the previous iteration. Therefore, if an upper bound f < n on the maximal flow value is known, the algorithm
    has the following complexities:
    O(fn^2) rounds, O(fn^4) communication, O(fn^4) computation.
    :param cap_matrix: The capacity matrix of the graph.
    :param source_selector: The selector array of the source.
    :param sink_selector: The selector array of the sink.
    :param max_flow_value: Optional: An upper bound on the maximal flow value in the graph.
    :param use_bfs_logarithmic_layered_graph: Whether to use the first part of the logarithmic BFS search to determine the layered subgraph. Defaults to True.
    :param bfs_algorithm: The BFS algorithm to be used to construct the layered graph. Ignored if use_bfs_logarithmic_layered_graph is True.
    :param timers: Whether debug timers should be enabled.
    :return: The flow graph of the identified maximal flow.
    """
    flow = cap_matrix.same_shape()
    flow.assign_all(0)

    n_nodes = cap_matrix.shape[0]

    iteration_bounds = [n_nodes]
    if max_flow_value is not None:
        iteration_bounds.append(max_flow_value)
    n_iterations = min(iteration_bounds)

    @for_range(n_iterations)
    def f(iter):
        residuals = cap_matrix.same_shape()
        residuals[:] = cap_matrix[:] - flow[:]

        full_adj_matrix = Matrix.create_from([
            [sint(residuals[i][j] > 0) for j in range(n_nodes)] for i in range(n_nodes)
        ])

        if use_bfs_logarithmic_layered_graph:
            # Find the layers and the layered graph.
            # O(log n) communication rounds, O(n^3 log n) communication, O(n^3 log n) computation.
            layer_selectors, layered_graph = bfs_logarithmic_layers_and_layered_subgraph(full_adj_matrix, source_selector, timers)

            # Filter layer with the sink such that it only contains the sink.
            # O(1) rounds, O(n^2) communication, O(n^2) computation.
            edges_to_keep = full_adj_matrix.same_shape()
            edges_to_keep.assign_all(1)
            for layer in layer_selectors:
                layer_has_sink = sint.dot_product(layer, sink_selector)
                nodes_to_remove = layer_has_sink * (layer[:] - sink_selector[:])
                for i in range(n_nodes):
                    edges_to_keep[i][:] -= nodes_to_remove
            layered_graph[:] *= edges_to_keep[:]

            # Now add the capacities.
            # O(1) rounds, O(n^2) communication, O(n^2) computation.
            layered_graph_capacities = cap_matrix.same_shape()
            layered_graph_capacities[:] = residuals[:] * layered_graph[:]
        else:
            _, bfs_matrix = bfs_algorithm(full_adj_matrix, source_selector, timers)
            # Because I originally wanted to trace the path from sink to source in the Edmonds-karp algorithm, the
            # bfs_matrix has edges from children to their parents in the bfs tree. For building layers, we need edges from
            # the parents to their children, therefore:
            bfs_matrix = bfs_matrix.transpose()

            # We build two things at the same time:
            # 1. The capacity matrix of the layered subgraph.
            # 2. "Selectors" for the nodes in each layer.
            # For the data-oblivious version of Tarjans algorithm to work correctly, the "last" layer, i.e., the layer containing
            # sink must only contain the sink, no other nodes, which we also enforce in this step.
            layered_graph_capacities = cap_matrix.same_shape()
            layered_graph_capacities.assign_all(0)

            layer_selectors = cap_matrix.same_shape()
            layer_selectors.assign_all(0)
            layer_selectors[0][:] = source_selector[:]

            @for_range(1, n_nodes)
            def layer_iter(i):
                last_layer = layer_selectors[i - 1]
                next_layer = sum([last_layer[i] * bfs_matrix[i][:] for i in range(n_nodes)])
                has_reached_sink = sint.dot_product(sink_selector, next_layer)
                next_layer += has_reached_sink * (sink_selector[:] - next_layer)
                layer_selectors[i][:] = next_layer

                for j in range(n_nodes):
                    for k in range(n_nodes):
                        layered_graph_capacities[j][k] += residuals[j][k] * last_layer[j] * next_layer[k]

        # print_ln("Layered adj matrix:")
        # for row in layered_graph_capacities:
        #     print_list(row)
        # print_ln("Layere selectors:")
        # for row in layer_selectors:
        #     print_list(row)
        blocking_flow = blocking_flow_tarjan(layered_graph_capacities, source_selector, sink_selector, layer_selectors)
        # print_ln("Blocking flow:")
        # for row in blocking_flow:
        #     print_list(row)

        flow[:] += blocking_flow[:]

    return flow

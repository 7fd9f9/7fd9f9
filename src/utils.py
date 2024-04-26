from typing import Callable

from Compiler.types import *
from Compiler.library import *
from Compiler.util import tree_reduce


UTILS_TIMER_OFFSET = 100_000
PATH_RECOVER_SAM_UPDATE = UTILS_TIMER_OFFSET + 1
PATH_RECOVER_SAM_SQUARE = UTILS_TIMER_OFFSET + 2
PATH_RECOVER_SAM_OVERFLOW_PREVENTION = UTILS_TIMER_OFFSET + 3

PATH_RECOVER_SAM_UPDATE_ITERATION_OFFSET = UTILS_TIMER_OFFSET + 1_000
PATH_RECOVER_SAM_SQUARE_ITERATION_OFFSET = UTILS_TIMER_OFFSET + 2_000
PATH_RECOVER_SAM_OVERFLOW_ITERATION_OFFSET = UTILS_TIMER_OFFSET + 3_000


def tree_select(data: list[any], op: Callable[[any, any], bool], mux: Callable[[sint, any, any], any] = sint.if_else) -> list[sint]:
    if len(data) == 1:
        # print_ln("n_sel: 1")
        return [cint(1)]
    # Phase 1. Tree-reduce
    next_layer = []
    decisions = []
    for i in range(0, len(data)-1, 2):
        left = data[i]
        right = data[i + 1]
        cmp = op(left, right)

        next_layer.append(mux(cmp, right, left))
        decisions.append(cmp)
    # print_ln("desc: " + "%s " * len(decisions), *[x.reveal() for x in decisions])
    if len(data) % 2 != 0:
        next_layer.append(data[-1])

    selector = tree_select(next_layer, op, mux)

    # Phase 2: Expanding selector
    new_selector = []
    for (old_selector, decision) in zip(selector, decisions):
        new_selector.append(old_selector * (1 - decision))
        new_selector.append(old_selector * decision)
    if len(data) % 2 != 0:
        new_selector.append(selector[-1])

    # print_ln("n_sel: " + "%s " * len(new_selector), *[x.reveal() for x in new_selector])
    return new_selector


def print_list(data: list[sint], name: str = None):
    print_ln((f"{name}: " if name is not None else "") + "%s " * len(data), *[x.reveal() for x in data])


def argmin(x):
    """ Compute index of maximum element.

    :param x: iterable
    :returns: sint or 0 if :py:obj:`x` has length 1
    """
    def op(a, b):
        comp = (a[1] < b[1])
        return comp.if_else(a[0], b[0]), comp.if_else(a[1], b[1])
    return tree_reduce(op, enumerate(x))[0]


def permute_adjacency_matrix(m: Matrix, permutation, reverse: bool = False) -> Matrix:
    res = m.same_shape()
    res[:] = m[:]
    res.secure_permute(permutation, reverse)
    res = res.transpose()
    res.secure_permute(permutation, reverse)
    res = res.transpose()
    return res


def clear_min(data: list[int | cint]) -> cint:
    fixed_ints = [x for x in data if isinstance(x, int)]
    if len(fixed_ints) == len(data):
        # All data-points are compile-time ints.
        # We quit now because Array.create_from fails otherwise.
        return cint(min(fixed_ints))

    # Find all cints.
    clear_ints = Array.create_from([x for x in data if isinstance(x, cint)])

    # Find a good starting point for the linear search, either the first cint or the minimal
    # compile-time value.
    result = clear_ints[0]
    if len(fixed_ints) > 0:
        result = cint(min(fixed_ints))

    # Linearly search for the minimum value.
    @for_range(len(clear_ints))
    def a(i):
        @if_(clear_ints[i] < result)
        def b():
            result.update(clear_ints[i])

    return result


def find_first_nonzero_columnwise(m: Matrix, compile_print: bool = False) -> Matrix:
    if compile_print:
        print(f"    Prefix sum.")

    # print_ln("Nodes after steps:")
    # for row in nodes_after_steps:
    #     print_list(row)

    rows = m.shape[0]
    columns = m.shape[0]

    prefix_sums = sint.Matrix(rows=rows - 1, columns=columns)
    prefix_sums[0][:] = m[0][:]
    for i in range(1, rows - 1):
        prefix_sums[i][:] = prefix_sums[i - 1][:] + m[i][:]
    break_point()

    if compile_print:
        print(f"        Done")
        print(f"    Selection.")

    # Name does not fit as well anymore, but requires fewer registers as creating a new matrix.
    prefix_sums[:] = prefix_sums[:] == 0

    result = m.same_shape()
    result[0][:] = m[0][:]
    for i in range(1, rows):
        result[i][:] = m[i][:] * prefix_sums[i-1][:]

    if compile_print:
        print(f"        Done")

    return result


def determine_path_matrix(parent_matrix: Matrix, end_node: Array, timers: bool = False, max_integer_value: int | None = None) -> Matrix:
    n_nodes = len(parent_matrix)

    node_after_steps = parent_matrix.same_shape()
    for i in range(n_nodes):
        node_after_steps[i][:] = end_node[:]
    max_values = [1 for _ in range(n_nodes)]

    current_square_adj = parent_matrix.same_shape()
    current_square_adj[:] = parent_matrix[:]
    current_square_adj_max_value = 1

    for i in range(math.ceil(math.log2(n_nodes))):
        # Perform eventual reduction steps to prevent integer overflows.
        # Square and multiply.
        if timers:
            start_timer(PATH_RECOVER_SAM_OVERFLOW_PREVENTION)
            start_timer(PATH_RECOVER_SAM_OVERFLOW_ITERATION_OFFSET + i)

        # if max_integer_value is not None:
        #
        #     if n_nodes * current_square_adj_max_value ** 2 > max_integer_value:
        #         print(f"Reducing square matrix to prevent integer overflows at power {i}.")
        #         # for x in range(n_nodes):
        #         #     for y in range(n_nodes):
        #         #         current_square_adj[x][y] = current_square_adj[x][y] != 0
        #         current_square_adj[:] = current_square_adj[:] != 0
        #         current_square_adj_max_value = 1
        #         break_point()
        #
        #     for j in range(n_nodes):
        #         if (j >> i) & 1 == 1:
        #             if (max_integer_value is not None and
        #                     n_nodes * max_values[j] * current_square_adj_max_value > max_integer_value):
        #                 print(f"Reducing after_steps {j} to prevent integer overflows in round {i}.")
        #                 # for k in range(n_nodes):
        #                 #     nodes_after_steps[j][k] = (nodes_after_steps[j][k] != 0)
        #                 node_after_steps[j][:] = node_after_steps[j][:] != 0
        #                 max_values[j] = 1

        if timers:
            stop_timer(PATH_RECOVER_SAM_OVERFLOW_ITERATION_OFFSET + i)
            stop_timer(PATH_RECOVER_SAM_OVERFLOW_PREVENTION)
            start_timer(PATH_RECOVER_SAM_UPDATE)
            start_timer(PATH_RECOVER_SAM_UPDATE_ITERATION_OFFSET + i)

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
            packed_nodes_after_steps[k][:] = node_after_steps[j][:]
        packed_nodes_after_steps[:] = packed_nodes_after_steps.direct_mul(current_square_adj)
        for k, j in enumerate(packing):
            node_after_steps[j][:] = packed_nodes_after_steps[k][:]

        if timers:
            stop_timer(PATH_RECOVER_SAM_UPDATE_ITERATION_OFFSET + i)
            stop_timer(PATH_RECOVER_SAM_UPDATE)
            stop_timer(PATH_RECOVER_SAM_SQUARE)
            start_timer(PATH_RECOVER_SAM_SQUARE_ITERATION_OFFSET + i)

        current_square_adj[:] = current_square_adj.direct_mul(current_square_adj)
        current_square_adj_max_value = n_nodes * current_square_adj_max_value ** 2

        if timers:
            stop_timer(PATH_RECOVER_SAM_SQUARE_ITERATION_OFFSET + i)
            stop_timer(PATH_RECOVER_SAM_SQUARE)

    break_point()

    # print_ln("Nodes after steps unnormalized:")
    # for row in node_after_steps:
    #     row.print_reveal_nested()

    # node_after_steps[:] = (node_after_steps[:] != 0)

    break_point()

    reversed_rows = node_after_steps.same_shape()
    for i in range(n_nodes):
        reversed_rows[i][:] = node_after_steps[n_nodes - i - 1][:]

    layered_adj_matrix_helper_matrix1 = sint.Matrix(rows=n_nodes, columns=n_nodes - 1)
    layered_adj_matrix_helper_matrix2 = sint.Matrix(rows=n_nodes - 1, columns=n_nodes)

    @for_range(n_nodes)
    def create_helper_matrix_outer(i: cint) -> None:
        @for_range(n_nodes - 1)
        def create_heler_matrix_inner(k: cint) -> None:
            layered_adj_matrix_helper_matrix1[i][k] = node_after_steps[n_nodes - k - 1][i]
            layered_adj_matrix_helper_matrix2[k][i] = node_after_steps[n_nodes - k - 2][i]

    result = parent_matrix.same_shape()
    result[:] = layered_adj_matrix_helper_matrix1.direct_mul(layered_adj_matrix_helper_matrix2)
    return result


def generate_graph(n_nodes: int) -> (Matrix, Array, Array):
    start_timer(1)
    adj_matrix = sint.Matrix(rows=n_nodes, columns=n_nodes)

    @for_range(n_nodes)
    def _(i):
        @for_range(n_nodes)
        def _(j):
            adj_matrix[i][j] = sint((i + j) % 2)


    start_selector = Array.create_from([sint(0) for _ in range(n_nodes)])
    start_selector[0] = sint(1)

    target_selector = Array.create_from([sint(0) for _ in range(n_nodes)])
    target_selector[-1] = sint(1)
    stop_timer(1)

    return adj_matrix, start_selector, target_selector




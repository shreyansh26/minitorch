from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    
    values = []
    for i, _ in enumerate(vals):
        if i == arg:
            values.append(f(vals[i] + epsilon))
        else:
            values.append(f(vals[i]))

    return [(values[i] - f(vals[i])) / epsilon for i in range(len(values))]

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sorted_vars = []
    visited = set()
    
    def dfs(v: Variable):
        if v.is_constant() or v in visited:
            return
        visited.add(v)
        if not v.is_leaf():
            for p in v.parents:
                dfs(p)
        sorted_vars.append(v)
    
    dfs(variable)
    return sorted_vars[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo_sorted_vars = topological_sort(variable)
    node_deriv = {variable.unique_id: deriv}

    for v in topo_sorted_vars:
        if v.unique_id not in node_deriv:
            continue
        
        curr_deriv = node_deriv[v.unique_id]
        chain_deriv = v.chain_rule(curr_deriv)

        for v, d in chain_deriv:
            if v.is_leaf():
                v.accumulate_derivative(d)
            else:
                if v.unique_id not in node_deriv:
                    node_deriv[v.unique_id] = d
                else:
                    node_deriv[v.unique_id] += d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

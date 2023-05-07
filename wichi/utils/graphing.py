from graphviz import Digraph
import numpy as np


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        if isinstance(n.data, np.ndarray):
            dot.node(name=uid, label=f"{{ {n.label}{' | ' if len(n.label)>0 else ''} data dim: {n.shape} | grad dim: {n.grad_shape} }}", shape='record')
        else:
            dot.node(name=uid, label=f"{{ {n.label}{' | ' if len(n.label)>0 else ''} data: {n.data:.4f} | grad: {n.grad:.4f} }}", shape='record')
        if n._op:
            # if this value is the result of some operation, create an op node for it
            dot.node(name=uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

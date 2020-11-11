import random
from graphviz import Digraph

from .operations import *

# TODO: replace with dictionary or even namedtuple to be more Pythonic
class Node:
    def __init__(self, name, func, id, placeholder=False):
        self.name = name
        self.func = func
        self.id = id
        self.open = True
        self.placeholder = placeholder

class ArchitectureSampler:
    def __init__(self, h_ops=None, com_ops=None, B=2, viz=False):
        self.B = B
        self._h_ops = h_ops
        self._com_ops = com_ops
        
        self.viz = viz
        self.g = Digraph('G', filename='sample.dot', engine='dot', format='png', node_attr={'shape': 'box'})

        self._hidden_state_set = [Node("input_h0", None, 0, placeholder=True), Node("input_h1", None, 1, placeholder=True)]
        self.node_counter = 2

    def _sample_hidden_state(self, block_num, k):
        out = random.sample(self._hidden_state_set, k=2)
        for item in out:
            if item.open is True:
                item.open = False
        return out

    def _sample_hidden_op(self, block_num):
        item_func = random.choice(self._h_ops)
        item = Node(item_func.__name__, item_func, self.node_counter)
        self.node_counter += 1
        return item

    def _sample_com_op(self, block_num):
        item_func = random.choice(self._com_ops)
        item = Node(item_func.__name__, item_func, self.node_counter)
        self.node_counter += 1
        return item

    def view_graph(self):
        self.g.view()
    
    def sample(self):
        cell_nodes = []
        subgraphs = []
        for i in range(self.B):
            # 1. and 2. select first hidden state from hidden state set
            h_a, h_b = self._sample_hidden_state(i, k=2)
            # 3. and 4. select operation to apply to h1 and h2
            h_a_op = self._sample_hidden_op(i)
            h_b_op = self._sample_hidden_op(i)
            assert(h_a_op.id != h_b_op.id)
            # 5. select operation to combine inputs
            com_op = self._sample_com_op(i)
            # Add com_op here to set of hidden states
            self._hidden_state_set.append(com_op)
            cell_nodes.append([h_a, h_b, h_a_op, h_b_op, com_op])
            # print([h_a, h_b, h_a_op, h_b_op, com_op])

            if self.viz:
                block_graph = Digraph(name=f"block{i}")
                if i == 0:
                    block_graph.attr('node', color='lightgrey', style='filled', rank='max')
                else:
                    block_graph.attr('node', color='green', style='filled', rank='max')
                block_graph.node(str(h_a.id), h_a.name)
                block_graph.node(str(h_b.id), h_b.name)
                block_graph.attr('node', color='yellow', style='filled')
                block_graph.node(str(h_a_op.id), h_a_op.name)
                block_graph.node(str(h_b_op.id), h_b_op.name)
                block_graph.attr('node', color='green', style='filled', rank='min')
                block_graph.node(str(com_op.id), com_op.name)
                block_graph.edge(str(h_a.id), str(h_a_op.id))
                block_graph.edge(str(h_b.id), str(h_b_op.id))
                block_graph.edge(str(h_a_op.id), str(com_op.id))
                block_graph.edge(str(h_b_op.id), str(com_op.id))
                self.g.subgraph(block_graph)
            
            # print([x.open for x in self._hidden_state_set])
        
        # look for all open hidden states and concat them depthwise
        unused_states = []
        for state in self._hidden_state_set:
            if state.open is True:
                unused_states.append(state)
        # print(unused_states)
        combine_operation = Node("concat", concat, self.node_counter)
        self.node_counter += 1
        combine_operation.id = "cell_final"
        cell_nodes.append(combine_operation)

        if self.viz:
            self.g.attr('node', color='pink', style='filled')
            self.g.node(combine_operation.id, combine_operation.name)
            for state in unused_states:
                self.g.edge(str(state.id), str(combine_operation.id))
        
        return cell_nodes

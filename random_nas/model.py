import torch
import torch.nn as nn

class NasCell(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, decisions):
        super(NasCell, self).__init__()
        self.stride = 1 if not reduction else 2
        self.B = len(decisions) - 1
        self.placeholders = [[None, None, None] for _ in range(self.B)]  # array which will store refs to data to hidden states used as inputs/outputs to blocks
        self.hidden_states = [None] * self.B                             # array which will store refs to hidden states used as inputs to blocks
        self.block_outputs = [None] * self.B                             # array which will store refs to outputs of blocks
        self.open_outputs = []                                           # array which will store refs to outputs of blocks that need to be concatenated together

        # torch modules themselves
        self.block_ops = []

        for i in range(self.B):
            block_nodes = decisions[i]  # [input_1, input_2, operation_1, operation_2, combine_operation]
            self.hidden_states[i] = [block_nodes[0].id, block_nodes[1].id]
            self.block_outputs[i] = block_nodes[4].id
            self.block_ops.append(nn.ModuleList([
                block_nodes[2].func(in_channels, out_channels, stride=self.stride),
                block_nodes[3].func(in_channels, out_channels, stride=self.stride),
            ]))
            self.combine_op = block_nodes[4].func
            if block_nodes[4].open is True:
                self.open_outputs.append(block_nodes[i].id)

        self.cell_modules = nn.ModuleList(self.block_ops)

    def forward(self, x, y):
        self.placeholders[0], self.placeholders[1] = x, y
        for i in range(self.B):
            # fetch input ids for block
            h_a = self.placeholders[self.hidden_states[i][0]]
            h_b = self.placeholders[self.hidden_states[i][1]]
            # forward pass through block
            h_a_out = self.block_ops[i][0](h_a)
            h_b_out = self.block_ops[i][1](h_b)
            # combine operation
            com_out = self.combine_op(h_a_out, h_b_out)
            print(self.block_outputs[i])
            self.placeholders[self.block_outputs[i]] = com_out
        # gather all open ends and concat together depthwise
        return torch.cat([self.placeholders[open_id] for open_id in self.open_outputs], dim=1)

class NasNet(nn.Module):
    def __init__(self, N, initial_filters, repeats, decisions):
        super(NasNet, self).__init__()
        self.first = True
        self.repeating_motifs = nn.ModuleList([self._create_motif(N, initial_filters * 2**i, decisions) for i in range(repeats)])

    def _create_motif(self, N, num_filters, decisions):
        # N normal cells followed by reduction cell
        if self.first:
            return nn.ModuleList([NasCell(3, num_filters, False, decisions)] + [NasCell(num_filters, num_filters, False, decisions) for _ in range(N)] + [NasCell(num_filters, num_filters, True, decisions)])
        else:
            return nn.ModuleList([NasCell(num_filters, num_filters, False, decisions) for _ in range(N)] + [NasCell(num_filters, num_filters, True, decisions)])

    # TODO: Figure out dealing with first NasCell input's residual connection
    def forward(self, x):
        for _, motif in enumerate(self.repeating_motifs):
            for i, layer in enumerate(motif):
                print("layer:")
                print(layer)
                if i == 0:
                    y = x
                    x = layer(y, x)
                else:
                    tmp = x
                    x = layer(y, x)
                    y = tmp
                    # TODO : make sure tmp not equal to x at this point.

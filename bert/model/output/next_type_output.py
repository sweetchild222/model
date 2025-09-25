import torch.nn as nn


class NextTypeOutput(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.linear = nn.Linear(hidden, 4)  #[is_next, is_not_next]
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax.forward(self.linear.forward(x[:, 0]))


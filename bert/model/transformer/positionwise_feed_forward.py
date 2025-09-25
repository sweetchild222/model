import torch.nn as nn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        x = self.w_1.forward(x)
        x = self.activation.forward(x)
        x = self.dropout.forward(x)
        x = self.w_2.forward(x)

        return x

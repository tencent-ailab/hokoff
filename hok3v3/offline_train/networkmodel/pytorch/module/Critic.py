import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DoubleMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DoubleMLPNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc3_v = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(state_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, action_dim)
        self.fc6_v = nn.Linear(hidden_dim, 1)

        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        bs, t, na, _ = X.shape
        X = X.reshape(-1, self.state_dim)
        norm_in_X = X

        h1 = self.nonlin(self.fc1(norm_in_X))
        h2 = self.nonlin(self.fc2(h1))
        a1 = self.fc3(h2)
        v1 = self.fc3_v(h2)
        out = (a1 + v1).reshape(bs, t, na, self.action_dim)

        h1_2 = self.nonlin(self.fc4(norm_in_X))
        h2_2 = self.nonlin(self.fc5(h1_2))
        a2 = self.fc3(h2_2)
        v2 = self.fc3_v(h2_2)
        out_2 = (a2 + v2).reshape(bs, t, na, self.action_dim)

        return out, out_2

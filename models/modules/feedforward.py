import torch

class Net(torch.nn.Module):
    def __init__(self, input_size, output_size=784,
                 positive_outputs=True, epsilon=1e-6, identity=False):
      super(Net, self).__init__()
      self.epsilon = epsilon
      self.linear1 = torch.nn.Linear(input_size, 1024)
      self.linear2 = torch.nn.Linear(1024, 1024)
      self.linear3 = torch.nn.Linear(1024, output_size)
      self.identity = identity
      self.positive_outputs = positive_outputs

    def forward(self, x):
      out = self.linear1(x)
      out = torch.nn.functional.relu(out)
      out = self.linear2(out)
      out = torch.nn.functional.relu(out)
      out = self.linear3(out)
      if self.identity:
        return 1 + 0 * out
      elif self.positive_outputs:
        return self.epsilon + torch.nn.functional.softplus(out)
      return out
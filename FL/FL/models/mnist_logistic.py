import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim = 1, output_dim = 10):

        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim*28*28, output_dim)
        self.device_ = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        #self.device = torch.device("cpu")
        self.to(self.device_)

        
    def forward(self, x):
        x = x.view(-1, 784)
        outputs = self.linear(x)
        return outputs

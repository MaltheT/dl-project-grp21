### FCv1

""" Hyperparameters """
BATCH_SIZE = 128
GAMMA = 0.95

exploration_rate = 1.
exploration_decay = 0.9995
exploration_min = 0.01

TARGET_UPDATE = 1000
learning_rate = 1e-4  
decay_rate = 0.99 
num_episodes = 10000
MEMORY_SIZE = 10000

class FC(nn.Module):    
    def __init__(self, outputs):
        super(FC, self).__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Linear(192, 512)
        self.h = nn.Linear(512, 512)
        self.output = nn.Linear(512, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.flatten(x)
        x = F.relu(self.input(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = self.output(x)
        return x

## FCv2

Same hyperparameters as v1

class FC(nn.Module):    
    def __init__(self, outputs):
        super(FC, self).__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Linear(192, 512)
        self.h = nn.Linear(512, 512)
        self.output = nn.Linear(512, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.flatten(x)
        x = F.relu(self.input(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = self.output(x)
        return x

## FCv3

class FC(nn.Module):    
    def __init__(self, outputs):
        super(FC, self).__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Linear(192, 256)
        self.h = nn.Linear(256, 256)
        self.output = nn.Linear(256, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.flatten(x)
        x = F.relu(self.input(x))
        x = F.relu(self.h(x))
        x = F.relu(self.h(x))
        x = self.output(x)
        return x
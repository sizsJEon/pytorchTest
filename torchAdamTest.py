# -*- coding: utf-8 -*-
import torch
import time

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N=128
print torch.__version__ 
print torch.cuda.is_available()
#device = torch.device("cuda:0")
device = torch.device("cpu")

print torch.cuda.get_device_name(0)

D_in, H1, H2, D_out = 10, 24, 12, 2

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)#, device=device)
y = torch.randn(N, D_out)#, device=device)

# Use the nn package to define our model and loss function.
sq = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
#    torch.nn.BatchNorm1d(H1),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
#    torch.nn.BatchNorm1d(H2),
    torch.nn.Linear(H2, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

#model=sq.cuda()
model=sq

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

sumV=0;
for t in range(500):
    start= time.time()    
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    end = time.time()
    sumV=sumV+(end-start)

    print '\tfTime:', (end-start), '|', (sumV/(t+1))

print sumV/500

# -*- coding: utf-8 -*-
import torch
import time

N=128
print torch.__version__ 
print torch.cuda.is_available()

device = torch.device("cuda:0")

print torch.cuda.get_device_name(0)

D_in, H1, H2, D_out = 10, 600, 300, 2

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

sq = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

model=sq.cuda()

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

sumV=0;
sumModelTime=0
sumLossTime=0
sumOptiTime=0
sumStepTime=0
for t in range(500):
    start= time.time()   
    lstart=start 

    y_pred = model(x)
#
    end=time.time()
    sumModelTime=sumModelTime+(end-start)
    print '\tmodelTime:', (sumModelTime/(t+1))
    start=end
#
    loss = loss_fn(y_pred, y)

#    if t % 100 == 99:
#        print(t, loss.item())        

    optimizer.zero_grad()

    loss.backward()
#
    end=time.time()
    sumLossTime=sumLossTime+(end-start)
    print '\tlossTime :', (sumLossTime/(t+1))
    start=end
#

    optimizer.step()

#
    end=time.time()
    sumOptiTime=sumOptiTime+(end-start)
    print '\toptiTime :', (sumOptiTime/(t+1))
    start=end
#


    end = time.time()
    sumV=sumV+(end-lstart)
    print '\tfTime:', (end-lstart), '|', (sumV/(t+1)),'\n'

print sumV/500

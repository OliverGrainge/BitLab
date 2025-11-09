import bitlab.bnn as bnn 
import torch 
import torch.optim as optim
import torch 


layer = bnn.BitLinear(128, 1, quant_type="none")

x = torch.randn(100, 128)
y = torch.randn(100, 1)


optimizer = torch.optim.SGD(layer.parameters(), lr=0.001)


for step in range(10000): 
    yhat = layer(x)
    loss = (y - yhat)**2 
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 250 == 0: 
        print(f"step: {step}, loss: {loss.item():.2f}")


    
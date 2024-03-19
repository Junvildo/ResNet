import torch
from models import *

input = torch.rand(1,3,224,224)

model = ResNet_50(num_class=1000)
output = model(input)

print(output.shape)
print(torch.argmax(torch.softmax(output, dim=1), dim=1))
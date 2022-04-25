import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LeNet_arch import LeNet5
import utils
import matplotlib.pyplot as plt


# check device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
rseed = 42
lrate = 0.001
bsize = 32
nepochs = 15

img_s = 32
nclass = 10

# define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
trdata = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valdata = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# define the data loaders
trloader = DataLoader(dataset=trdata, 
                          batch_size=bsize, 
                          shuffle=True)

valloader = DataLoader(dataset=valdata, 
                          batch_size=bsize, 
                          shuffle=False)
hr = 10
vr = 5

fig = plt.figure()
for idx in range(1, hr * vr + 1):
    plt.subplot(vr, hr, idx)
    plt.axis('off')
    plt.imshow(trdata.data[idx], cmap='gray_r')
fig.suptitle('MNIST Dataset - preview');

plt.show()

torch.manual_seed(rseed)

mdl = LeNet5(nclass).to(dev)
opt = torch.optim.Adam(mdl.parameters(), lr=lrate)
crit = nn.CrossEntropyLoss()

mdl, opt, _ = utils.training_loop(mdl, crit, opt, trloader, valloader, nepochs, dev)

PATH = "lenet5_mdl.pt"
torch.save(mdl, PATH)

#predictions
PATH = "lenet5_mdl.pt"
model = torch.load(PATH)
model.eval()

hr = 10
vr = 5

fig = plt.figure()
for idx in range(1, hr * vr + 1):
    plt.subplot(vr, hr, idx)
    plt.axis('off')
    plt.imshow(valdata.data[idx], cmap='gray_r')
    
    with torch.no_grad():
        model.eval()
        _, probs = model(valdata[idx][0].unsqueeze(0))
        
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
    
    plt.title(title, fontsize=7)
fig.suptitle('LeNet-5 - predictions');
plt.show()
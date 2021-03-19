From Me to Everyone:  05:26 PM
https://discuss.pytorch.org/t/custom-loss-kl-divergence-error/19850/5
From Me to Everyone:  05:40 PM
https://medium.com/@sikdar_sandip/implementing-a-variational-autoencoder-vae-in-pytorch-4029a819ccb6
https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
From Pranav A Pillai to Everyone:  06:37 PM
def KL(mu, sigma):
  runningKL = 0
  for i in range(8):
    contribution = 1 + torch.log(sigma[i]**2) - mu[i]**2 - sigma[i]**2
    runningKL += -torch.sum(contribution)/2
  return runningKL/mu.shape[0]
def AELoss(x, mu, sigma, y):
  kl_loss = KL(mu, sigma)
  bce = nn.BCELoss()(y, x)
  # print('kl: ', kl_loss, 'bce: ', bce)
  return kl_loss + bce
lr = 1e-3
np.random.seed(20)

encoder = Encoder()
decoder = Decoder()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

train_error_list = []
train_loss_list = []
val_error_list = []
val_loss_list = []

# train for at least 1000 iterations
for t in range(100):
    # 1. sample a mini-batch of size bb = 32
    x, y = train_dataloader.__next__()

    x = torch.Tensor(x)
    y = torch.Tensor(y).long()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoding = encoder(x)
    mu = encoding[:, :8]
    sigma = encoding[:, 8:]
    decoding = decoder(mu)

    total_loss = AELoss(x.reshape((x.shape[0], -1)), mu, sigma, decoding)
    total_loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    print(t, total_loss.item())
    train_loss_list.append(total_loss.item())
From Pranav A Pillai to Everyone:  06:51 PM
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        reshaped = x.reshape(x.shape[0], -1)
        out = torch.tanh(self.fc1(reshaped))
        out = self.fc2(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 196)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.sigmoid(self.fc2(out))
        return out
fig, axs = plt.subplots(2, 4)
index = np.random.randint(len(valX), size=8)
i = 0

num_samples = 4

# for r in range(2):
for c in range(num_samples):
    x = valX_resized[index[i]]
    axs[0, c].imshow(x, cmap=plt.cm.gray)
    axs[0, c].set_title(str(valY[index[i]]))
    axs[0, c].axis('off')
    
    x = torch.Tensor(x).unsqueeze(0)
    encoding = encoder(x)
    mu = encoding[:, :8]
    sigma = encoding[:, 8:]
    decoding = decoder(mu).reshape((14, 14)).detach().numpy()

    axs[1, c].imshow(decoding, cmap=plt.cm.gray)
    axs[1, c].set_title(str(valY[index[i]]))
    axs[1, c].axis('off')
    i+=1

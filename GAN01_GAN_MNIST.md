**Using a GAN with 2 FC layers to generate MINST Images**


```python
import numpy as np
import pandas as pd
import plotly.express as px

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as T

from tqdm import trange, tqdm

from IPython.display import clear_output
```


```python
img_size = 28
img_channels = 1
z_dim = 100

n_hidden = 128
n_classes = 10

n_epochs = 50
batch_size = 100
lr = 2e-4
k= 20 # G_lr/D_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```


```python
train_dataset = MNIST(root="./data", train=True, download=True, transform=T.Compose(
        [
            T.ToTensor(), 
            T.Normalize((0.5,), (0.5,)),
        ]
    ))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

n_samples = len(train_loader.dataset) # type: ignore
n_batches = len(train_loader)
print(f"Number of training samples: {n_samples}")
print(f"Number of batches: {n_batches}")
```


```python
imgs, labels = next(iter(train_loader))
print(f"Image batch shape: {imgs.shape}")
print(f"Label batch shape: {labels.shape}")
```


```python
def denormalize(imgs):
    return imgs * 0.5 + 0.5

def show_images(imgs, grid_size=5):
    imgs = denormalize(imgs).squeeze(1).numpy()
    num_imgs = min(len(imgs), grid_size*grid_size)
    imgs = imgs[:num_imgs]

    img_h, img_w = imgs.shape[1], imgs.shape[2]
    # 拼成 grid
    grid_imgs = np.block([[imgs[i*grid_size + j] for j in range(grid_size)] for i in range(grid_size)])

    fig = px.imshow(grid_imgs, color_continuous_scale='gray', aspect='')
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(title="MNIST Images", coloraxis_showscale=False)
    fig.show()

show_images(imgs)
```

![image.png](https://cdn.jsdelivr.net/gh/KuiMian/NoteImage@master/2025/10/upgit_20251004_show_data.png)


```python
class Generator(nn.Module):
    def __init__(self, z_dim, n_hidden, img_size, img_channels):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(n_hidden, img_size*img_size*img_channels),
            nn.Tanh()
        )
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, z):
        return self.net(z).view(-1, self.img_channels, self.img_size, self.img_size)

class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels, n_hidden):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size*img_size*img_channels, n_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, x):
        x = x.view(-1, self.img_size*self.img_size*self.img_channels)
        return self.net(x)
```


```python
def train_D(D, D_optimizer, criterion, real_imgs, fake_imgs):

    real_labels = torch.ones(real_imgs.size(0), 1).to(device)
    fake_labels = torch.zeros(fake_imgs.size(0), 1).to(device)

    real_preds = D(real_imgs)
    real_loss = criterion(real_preds, real_labels)
    fake_preds = D(fake_imgs.detach())
    fake_loss = criterion(fake_preds, fake_labels)

    D_loss = real_loss + fake_loss
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item(), torch.mean(real_preds).item(), torch.mean(fake_preds).item()

def train_G(G, D, G_optimizer, criterion, fake_imgs):
    real_labels = torch.ones(fake_imgs.size(0), 1).to(device)

    preds = D(fake_imgs)
    G_loss = criterion(preds, real_labels)

    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

def fit(D, G, D_optimizer: torch.optim.Adam, G_optimizer: torch.optim.Adam, scheduler, criterion, n_epochs):

    train_history = {
        "D_loss": [],
        "G_loss": [],
        "D_real": [],
        "D_fake": [],
        "lr": [],
    }

    for epoch in trange(n_epochs, desc="Epoch"):
        D_loss, G_loss, real_score, fake_score = 0.0, 0.0, 0.0, 0.0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(device)

            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = G(z)

            _D_loss, _real_score, _fake_score = train_D(D, D_optimizer, criterion, real_imgs, fake_imgs)
            D_loss += _D_loss
            real_score += _real_score
            fake_score += _fake_score

            _G_loss = train_G(G, D, G_optimizer, criterion, fake_imgs)
            G_loss += _G_loss

        train_history["D_loss"].append(D_loss / n_samples)
        train_history["G_loss"].append(G_loss / n_samples)
        train_history["D_real"].append(real_score / n_batches)
        train_history["D_fake"].append(fake_score / n_batches)
        train_history["lr"].append(scheduler.get_last_lr()[0])

        G_optimizer.param_groups[0]['lr'] = train_history["lr"][-1] * k

        if epoch == 0 or (epoch+1) % 10 == 0:
            clear_output(wait=True)
            
            tqdm.write(
                f"Epoch [{epoch+1}/{n_epochs}] lr: {train_history['lr'][-1]:.6f} "
                f"D_loss: {train_history['D_loss'][-1]:.4f} G_loss: {train_history['G_loss'][-1]:.4f} "
                f"D_real: {train_history['D_real'][-1]:.4f} D_fake: {train_history['D_fake'][-1]:.4f} "
            )

            z = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = G(z).detach().cpu()
            show_images(fake_imgs)

        scheduler.step()


    return train_history
```


```python
D = Discriminator(img_size, img_channels, n_hidden).to(device)
G = Generator(z_dim, n_hidden, img_size, img_channels).to(device)

optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
optim_G = torch.optim.Adam(G.parameters(), lr=lr*k, betas=(0.5, 0.999))

scheduler = lr_scheduler.LambdaLR(optim_D, lr_lambda=lambda epoch: np.exp(-0.1*epoch))
criterion = nn.BCELoss(reduction='sum')

train_history = fit(D, G, optim_D, optim_G, scheduler, criterion, n_epochs)
```

![last_epoch](https://cdn.jsdelivr.net/gh/KuiMian/NoteImage@master/2025/10/upgit_20251004_last_epoch.png)




```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss", "D(x) & D(G(z))", "Learning Rate"))

fig.add_traces([
    go.Scatter(y=train_history["D_loss"], mode='lines', name='D_loss'),
    go.Scatter(y=train_history["G_loss"], mode='lines', name='G_loss'),
], rows=1, cols=1)

fig.add_traces([
    go.Scatter(y=train_history["D_real"], mode='lines', name='D(x)'),
    go.Scatter(y=train_history["D_fake"], mode='lines', name='D(G(z))'),
], rows=1, cols=2)

fig.add_hline(
    y=0.5,
    line=dict(color="black", dash="dash"),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(y=train_history["lr"], mode='lines', name='Learning Rate'),
    row=1, col=3
)

fig.update_layout(height=400, width=1200, title_text="GAN Training History")
fig.show()

```

![train_history](https://cdn.jsdelivr.net/gh/KuiMian/NoteImage@master/2025/10/upgit_20251004_train_history.png)|

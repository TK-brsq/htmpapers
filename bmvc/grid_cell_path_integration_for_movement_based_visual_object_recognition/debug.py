import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== Parameters =====
CROP_X, CROP_Y = 14, 14   # crop center
CROP_H, CROP_W = 20, 20   # crop size
RESIZE = 32                 # resize to RESIZE x RESIZE
LATENT_DIM = 64            # d-dimensional vector
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

# ===== 1. Download MNIST =====
raw_transform = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=raw_transform)
mnist_test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=raw_transform)

# ===== 2 & 3. Custom transform: Crop -> Resize =====
class CropAndResize:
    """Center-based crop then resize."""
    def __init__(self, cx, cy, h, w, size):
        self.cx, self.cy = cx, cy
        self.h, self.w = h, w
        self.size = size
        self.resize = transforms.Resize((size, size))

    def __call__(self, img_tensor):
        # img_tensor: (1, H, W)
        _, H, W = img_tensor.shape
        x1 = max(self.cx - self.w // 2, 0)
        y1 = max(self.cy - self.h // 2, 0)
        x2 = min(x1 + self.w, W)
        y2 = min(y1 + self.h, H)
        cropped = img_tensor[:, y1:y2, x1:x2]       # (1, h, w)
        resized  = self.resize(cropped)               # (1, size, size)
        return resized

pipeline = transforms.Compose([
    transforms.ToTensor(),
    CropAndResize(CROP_X, CROP_Y, CROP_H, CROP_W, RESIZE),
])

mnist_train.transform = pipeline
mnist_test.transform  = pipeline

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(mnist_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===== 4. CNN Autoencoder =====
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # 1 x 32 x 32 -> 32 x 16 x 16 -> 64 x 8 x 8 -> 128 x 4 x 4 -> flatten -> latent
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # After 3x stride-2 conv: 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)          # (B, latent_dim)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d(32,   1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        return self.deconv(h)      # (B, 1, 32, 32)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ===== Training =====
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ConvAutoencoder(LATENT_DIM).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            loss = loss_fn(recon, imgs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(mnist_train)
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  Recon Loss: {avg:.6f}")

    # ===== Feature Extraction (inference) =====
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            z = model.encoder(imgs)          # (B, LATENT_DIM)
            all_features.append(z.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)  # (N_test, LATENT_DIM)
    labels   = torch.cat(all_labels,   dim=0)

    print(f"\nExtracted feature shape: {features.shape}")  # e.g. torch.Size([10000, 64])
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.utils import save_image

# ===== Parameters =====
CROP_CENTERS = [(4, 4), (8, 8), (12, 12)]  # list of (cx, cy)
CROP_H, CROP_W = 16, 16
RESIZE = 32
LATENT_DIM = 64
BATCH_SIZE = 128
EPOCHS = 1
LR = 1e-3

# ===== 1. Download MNIST =====
raw_transform = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=raw_transform)
mnist_test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=raw_transform)

# ===== 2 & 3. Multi-crop Dataset Wrapper =====
class CropAndResize:
    def __init__(self, cx, cy, h, w, size):
        self.cx, self.cy = cx, cy
        self.h, self.w = h, w
        self.resize = transforms.Resize((size, size))

    def __call__(self, img_tensor):
        _, H, W = img_tensor.shape

        # クロップ領域の座標（元画像座標系）
        x1 = self.cx - self.w // 2
        y1 = self.cy - self.h // 2
        x2 = x1 + self.w
        y2 = y1 + self.h

        # 元画像からクリップした有効領域
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, W), min(y2, H)

        # 出力バッファ（黒で初期化）
        out = torch.zeros(1, self.h, self.w, dtype=img_tensor.dtype)

        # パディング後バッファ内での貼り付け先オフセット
        dst_x = x1c - x1
        dst_y = y1c - y1

        out[:, dst_y:dst_y + (y2c - y1c), dst_x:dst_x + (x2c - x1c)] = \
            img_tensor[:, y1c:y2c, x1c:x2c]

        return self.resize(out)


class MultiCropMNIST(Dataset):
    def __init__(self, base_dataset, crop_centers, crop_h, crop_w, size):
        self.base = base_dataset
        self.croppers = [CropAndResize(cx, cy, crop_h, crop_w, size) for cx, cy in crop_centers]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]   # img: (1, H, W) tensor
        # crops: (n, 1, size, size)
        crops = torch.stack([c(img) for c in self.croppers], dim=0)
        return crops, label

raw_mnist_train = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transforms.ToTensor())
raw_mnist_test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dataset = MultiCropMNIST(raw_mnist_train, CROP_CENTERS, CROP_H, CROP_W, RESIZE)
test_dataset  = MultiCropMNIST(raw_mnist_test,  CROP_CENTERS, CROP_H, CROP_W, RESIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===== 4. CNN Autoencoder =====
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)

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
        return self.deconv(self.fc(z).view(-1, 128, 4, 4))

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode_multicrop(self, crops):
        """
        crops: (B, n, 1, size, size)
        returns: (B, n, latent_dim)
        """
        B, n, C, H, W = crops.shape
        flat = crops.view(B * n, C, H, W)          # (B*n, 1, H, W)
        z = self.encoder(flat)                      # (B*n, latent_dim)
        return z.view(B, n, -1)                     # (B, n, latent_dim)

def save_debug_images(
    raw_dataset,       # transforms.ToTensor() 済みの生MNISTデータセット
    crop_dataset,      # MultiCropMNIST（リサイズ済みクロップを返す）
    crop_centers,      # [(cx, cy), ...]
    crop_h, crop_w,    # クロップサイズ（リサイズ前）
    n_samples=1,
    save_dir="debug_crops",
):
    os.makedirs(save_dir, exist_ok=True)
    n_crops = len(crop_centers)

    for i in range(n_samples):
        orig, label = raw_dataset[i]          # (1, 28, 28)
        crops_resized, _ = crop_dataset[i]    # (n, 1, 32, 32)

        sample_dir = os.path.join(save_dir, f"sample{i:03d}_label{label}")
        os.makedirs(sample_dir, exist_ok=True)

        # --- (a) 元画像 ---
        save_image(orig, os.path.join(sample_dir, "orig.png"))

        for j, (cx, cy) in enumerate(crop_centers):
            _, H, W = orig.shape

            # --- (b) クロップ前の座標（パディング込み） ---
            x1 = cx - crop_w // 2
            y1 = cy - crop_h // 2
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, W), min(y2, H)

            # 黒バッファに貼り付け（CropAndResizeと同じロジック）
            crop_raw = torch.zeros(1, crop_h, crop_w, dtype=orig.dtype)
            dst_x, dst_y = x1c - x1, y1c - y1
            crop_raw[:, dst_y:dst_y + (y2c - y1c), dst_x:dst_x + (x2c - x1c)] = \
                orig[:, y1c:y2c, x1c:x2c]

            save_image(crop_raw, os.path.join(sample_dir, f"crop{j}_cx{cx}_cy{cy}_raw.png"))

            # --- (c) リサイズ済みクロップ（Datasetから取得） ---
            save_image(crops_resized[j], os.path.join(sample_dir, f"crop{j}_cx{cx}_cy{cy}_resized.png"))

    print(f"Saved {n_samples} samples to '{save_dir}/'")
    print("Structure: sample<i>_label<L>/ orig.png | crop<j>_*_raw.png | crop<j>_*_resized.png")


# ===== Training =====
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = ConvAutoencoder(LATENT_DIM).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n = len(CROP_CENTERS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for crops, _ in train_loader:
            crops = crops.to(device)               # (B, n, 1, 32, 32)
            B = crops.size(0)
            flat = crops.view(B * n, 1, RESIZE, RESIZE)   # (B*n, 1, 32, 32)

            recon, _ = model(flat)
            loss = loss_fn(recon, flat)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * B
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  Recon Loss: {total_loss/len(train_dataset):.6f}")

    # ===== Feature Extraction: shape [データ数, n, latent_dim] =====
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for crops, labels in test_loader:
            crops = crops.to(device)               # (B, n, 1, 32, 32)
            z = model.encode_multicrop(crops)      # (B, n, latent_dim)
            all_features.append(z.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)      # (N_test, n, latent_dim)
    labels   = torch.cat(all_labels,   dim=0)

    print(f"\nExtracted feature shape: {features.shape}")
    # -> torch.Size([10000, 3, 64])  ※ n=3, latent_dim=64 の場合

    save_debug_images(
        raw_dataset   = raw_mnist_test,
        crop_dataset  = test_dataset,
        crop_centers  = CROP_CENTERS,
        crop_h        = CROP_H,
        crop_w        = CROP_W,
        n_samples     = 1,
        save_dir      = "debug_crops",
    )

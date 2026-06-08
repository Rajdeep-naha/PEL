import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import csv

# GPU ACCELERATION & AUDIO
import kornia.augmentation as K
import torchaudio.transforms as AT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join("/tmp", f"triton-cache-{os.getuid()}"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join("/tmp", f"torchinductor-cache-{os.getuid()}"))

from pel.config import *
from pel.data_loader import get_vision_loaders, get_audio_loaders, get_text_loaders
from pel.models import get_resnet_encoder, TextPeLEncoder, SimCLRProjector
from pel.utils import set_seed

# Hardware Tuning
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ==========================================
# 🧠 LOSS & HELPERS
# ==========================================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        z_i, z_j = F.normalize(z_i.float(), dim=1), F.normalize(z_j.float(), dim=1)
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        sim = sim.masked_fill(mask, -1e4) 
        labels = torch.arange(2 * batch_size, device=z_i.device)
        labels[:batch_size] += batch_size
        labels[batch_size:] -= batch_size
        return self.criterion(sim, labels)

def save_checkpoint(model, optimizer, epoch, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'optimizer_state_dict': optimizer.state_dict()}, path)

def load_checkpoint_for_resume(model, optimizer, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        print(f"    🔄 Resuming from checkpoint: {filename}")
        ckpt = torch.load(path, map_location=DEVICE)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_dict = ckpt["model_state_dict"]

        # Older text checkpoints used a projector with BatchNorm:
        # net = Linear -> BatchNorm1d -> ReLU -> Linear.
        # The current projector is Linear -> ReLU -> Linear.
        # Remap the final linear layer and drop BatchNorm-only keys.
        if "net.3.weight" in state_dict and "net.2.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["net.2.weight"] = state_dict.pop("net.3.weight")
            state_dict["net.2.bias"] = state_dict.pop("net.3.bias")
            for key in [
                "net.1.weight",
                "net.1.bias",
                "net.1.running_mean",
                "net.1.running_var",
                "net.1.num_batches_tracked",
            ]:
                state_dict.pop(key, None)

        missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    ⚠️ Resume load missing keys: {missing[:5]}")
        if unexpected:
            print(f"    ⚠️ Resume load unexpected keys: {unexpected[:5]}")

        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            print(f"    ⚠️ Skipping optimizer resume due to state mismatch: {exc}")
        return ckpt['epoch'] + 1
    return 0

def log_to_csv(filename, epoch, loss):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", filename)
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["epoch", "loss"])
        writer.writerow([epoch, loss])

# ==========================================
# 🎨 AUDIO TRANSFORM (Mel-Spectrogram + SpecAugment)
# ==========================================
class SimCLRAudioTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def apply_single_hard_augment(self, x):
        """Applied to each sample individually to ensure high training quality."""
        # x: [1, Samples]
        orig_len = x.shape[-1]
        
        # 1. 🏆 Temporal Random Resized Crop (Per-sample unique)
        crop_ratio = torch.empty(1).uniform_(0.7, 0.95).item()
        crop_len = int(orig_len * crop_ratio)
        start = torch.randint(0, orig_len - crop_len + 1, (1,))
        
        x_aug = x[:, start:start + crop_len]
        # Resample back to original length
        x_aug = F.interpolate(x_aug.unsqueeze(0), size=orig_len, mode='linear', align_corners=False).squeeze(0)

        # 2. 🔊 Individual Noise
        noise = 0.012 * torch.randn_like(x_aug) + 0.005 * torch.rand_like(x_aug)
        x_aug = x_aug + noise

        # 3. 📉 Random Gain
        gain = torch.empty(1).uniform_(0.4, 1.6).to(x.device)
        x_aug = x_aug * gain

        # 4. ✂️ Multiple Time Masks (Individualized)
        for _ in range(2):
            mask_len = int(orig_len * torch.empty(1).uniform_(0.05, 0.12))
            mask_start = torch.randint(0, orig_len - mask_len, (1,))
            x_aug[:, mask_start:mask_start + mask_len] = 0

        # 5. 🔄 Polarity Inversion
        if torch.rand(1) > 0.5:
            x_aug = -x_aug

        return x_aug

    def forward(self, x):
        # x: [Batch, 1, Samples]
        # We use torch.stack to ensure every sample in the batch is different
        x_i = torch.stack([self.apply_single_hard_augment(t) for t in x])
        x_j = torch.stack([self.apply_single_hard_augment(t) for t in x])
        return x_i, x_j
# ==========================================
# 👂 AUDIO TRACK
# ==========================================
def train_audio(args):
    print(f"\n👂 STARTING 200-EPOCH AUDIO PRE-TRAINING (M11)")
    print(f"🔥 Mode: Hard Augmentations | Temp: {SIMCLR_TEMPERATURE}")
    
    train_loader, _ = get_audio_loaders(stage='pretrain')
    
    # Encoder and Projector
    from pel.models import M11
    encoder = M11(n_input=1, n_output=AUDIO_NUM_CLASSES)
    model = SimCLRProjector(encoder, 128).to(DEVICE)
    
    gpu_aug = SimCLRAudioTransform().to(DEVICE) 
    
    # 200 Epochs requires robust optimization
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=AUDIO_LR_PRETRAIN, weight_decay=1e-4)
    
    # Scheduler strictly tuned for 200 epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=AUDIO_LR_PRETRAIN, 
        steps_per_epoch=len(train_loader), 
        epochs=200,
        pct_start=0.1 # 10% warmup, 90% decay
    )
    
    criterion = NTXentLoss(temperature=SIMCLR_TEMPERATURE) # Use 0.1 from config

    ckpt_name = PRETRAINED_CHECKPOINTS["audio"]
    start_epoch = load_checkpoint_for_resume(model, optimizer, ckpt_name)

    for epoch in range(start_epoch, AUDIO_EPOCHS_PRETRAIN):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/200")
        
        for x, _ in loop:
            x = x.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Unique hard views for every sample in batch
                x_i, x_j = gpu_aug(x)
                
                z_i, z_j = model(x_i), model(x_j)
                loss = criterion(z_i, z_j)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- 💾 LOGGING & CHECKPOINTING ---
        avg_loss = total_loss / len(train_loader)
        log_to_csv("audio_pretraining_m11.csv", epoch + 1, avg_loss)
        
        # Save latest
        save_checkpoint(model, optimizer, epoch, ckpt_name)
        
        # Save milestones every 50 epochs to prevent data loss
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch, f"pel_audio_m11_ep{epoch+1}.pth")

    print(f"✅ Training Complete. Final weights saved to {ckpt_name}")

# ==========================================
# 👁️ VISION TRACK (Unchanged)
# ==========================================
def train_vision(args):
    print("\n👁️  STARTING VISION PRE-TRAINING...")
    train_loader, _ = get_vision_loaders(stage="pretrain")
    encoder, dim = get_resnet_encoder(modality='vision') 
    model = SimCLRProjector(encoder, dim).to(DEVICE)
    model = torch.compile(model)
    
    gpu_aug = nn.Sequential(
        # 1. Scale invariant cropping (forces model to see parts of objects)
        K.RandomResizedCrop(size=(VISION_IMG_SIZE, VISION_IMG_SIZE), scale=(0.08, 1.0), p=1.0),
        
        # 2. Horizontal flip (standard)
        K.RandomHorizontalFlip(p=0.5),
        
        # 3. Aggressive Color Jitter (prevents "cheating" via color histograms)
        # Strength increased from 0.4 to 0.8/1.0
        K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        
        # 4. Random Grayscale (forces model to learn shapes, not just colors)
        K.RandomGrayscale(p=0.2),
        
        # 5. Gaussian Blur (removes high-frequency noise shortcuts)
        # Kernel size should be ~10% of image size (e.g., 23 for 224x224)
        K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=0.5),
        
        # 6. Final Normalization (MUST be only normalization in the pipeline)
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=VISION_LR_PRETRAIN, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=VISION_LR_PRETRAIN, steps_per_epoch=len(train_loader), epochs=VISION_EPOCHS_PRETRAIN)
    criterion = NTXentLoss(temperature=0.2)

    ckpt_name = "pel_vision_full.pth"
    start_epoch = load_checkpoint_for_resume(model, optimizer, ckpt_name)

    for epoch in range(start_epoch, VISION_EPOCHS_PRETRAIN):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for x, _ in loop:
            x = x.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                x_i, x_j = gpu_aug(x), gpu_aug(x)
                z_i, z_j = model(x_i), model(x_j)
                loss = criterion(z_i, z_j)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        log_to_csv("vision_pretraining.csv", epoch + 1, total_loss/len(train_loader))
        save_checkpoint(model, optimizer, epoch, ckpt_name)

# ==========================================
# 📚 TEXT TRACK (Unchanged)
# ==========================================
def train_text(args):
    print(f"\n📚 STARTING TEXT PRE-TRAINING...")
    train_loader, _ = get_text_loaders(stage="pretrain")
    encoder = TextPeLEncoder(pad_idx=1) 
    model = SimCLRProjector(encoder, TEXT_EMBED_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=TEXT_EPOCHS_PRETRAIN)
    criterion = NTXentLoss(temperature=0.1)

    ckpt_name = "pel_text_full.pth"
    start_epoch = load_checkpoint_for_resume(model, optimizer, ckpt_name)

    for epoch in range(start_epoch, TEXT_EPOCHS_PRETRAIN):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for x_i, x_j in loop:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z_i, z_j = model(x_i), model(x_j)
                loss = criterion(z_i, z_j)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        log_to_csv("text_pretraining.csv", epoch + 1, total_loss/len(train_loader))
        save_checkpoint(model, optimizer, epoch, ckpt_name)

if __name__ == "__main__":
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, choices=['vision', 'audio', 'text'])
    args = parser.parse_args()

    if args.modality == 'vision': train_vision(args)
    elif args.modality == 'audio': train_audio(args)
    elif args.modality == 'text': train_text(args)

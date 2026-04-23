"""
Self-Pruning Neural Network — Tredence Analytics Case Study
============================================================
A feed-forward network that learns to prune its own weights *during* training
via learnable gate parameters and L1 sparsity regularisation.

Structure
---------
  Part 1 : PrunableLinear  — gated linear layer
  Part 2 : sparsity_loss   — L1 regularisation term
  Part 3 : Training loop   — three λ experiments on CIFAR-10 + report
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 – PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable gate before the forward pass.

    Gate mechanism
    --------------
    gates = sigmoid(gate_scores)          ∈ (0, 1)  — one scalar per weight
    pruned_weights = weight * gates
    output = pruned_weights @ x^T + bias

    Because sigmoid is differentiable everywhere, gradients flow through
    `gates` back to `gate_scores`, so the optimiser can drive gate_scores
    toward -∞ (gate → 0, weight is pruned) or +∞ (gate → 1, weight is kept).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard affine parameters (shape: out × in)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — registered as a parameter so the optimiser updates them.
        # Same shape as weight: one gate per individual weight element.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Kaiming uniform for weights — identical to nn.Linear's default init.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Start gate_scores at 0 → sigmoid(0) = 0.5.
        # Neutral starting point so the L1 penalty can close weak gates quickly.
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Squash gate_scores into (0,1). Gradients flow through sigmoid.
        gates = torch.sigmoid(self.gate_scores)

        # 2. Element-wise gating — a gate of 0 completely zeros the weight.
        pruned_weights = self.weight * gates

        # 3. Standard affine transform using gated weights.
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current gate values as a detached tensor (for analysis)."""
        return torch.sigmoid(self.gate_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Three-layer feed-forward network built entirely from PrunableLinear layers.

    CIFAR-10: 3 × 32 × 32 = 3 072 input pixels → 10 output classes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # (B, 3 × 32 × 32) → (B, 3072)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ── Part 2 helpers ────────────────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        Since gates = sigmoid(gate_scores) are always positive, their L1 norm
        is simply their sum.  Adding λ * this to the cross-entropy loss creates
        a gradient that pushes gate_scores toward -∞ (gate → 0).
        """
        total = torch.zeros(1, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total = total + torch.sigmoid(m.gate_scores).sum()
        return total

    @torch.no_grad()
    def sparsity_level(self, threshold: float = 0.1) -> float:
        """Percentage of weights whose gate value is below `threshold`."""
        total = pruned = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                total += gates.numel()
                pruned += (gates < threshold).sum().item()
        return 100.0 * pruned / total if total > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """Flat numpy array of all gate values (used for plotting)."""
        parts = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                parts.append(torch.sigmoid(m.gate_scores).cpu().numpy().ravel())
        return np.concatenate(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()   # pin_memory only helps on CUDA
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 – Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparsity: float,
    device: torch.device,
) -> tuple[float, float]:
    """One full pass over the training set. Returns (avg_loss, accuracy %)."""
    model.train()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # ── Part 2: Total Loss = CrossEntropyLoss + λ * SparsityLoss ─────────
        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lambda_sparsity * sp_loss

        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    """Return test accuracy (%)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total   += images.size(0)
    return 100.0 * correct / total


def run_experiment(
    lambda_sparsity: float,
    epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple["SelfPruningNet", float, float]:
    """Train one model with the given λ and return (model, test_acc, sparsity)."""
    print(f"\n{'='*60}")
    print(f"  Experiment  λ = {lambda_sparsity}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(device)
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": 1e-3},
        {"params": gate_params,   "lr": 5e-2},   # gates need a higher lr to converge fast
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, lambda_sparsity, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            sparsity = model.sparsity_level()
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"sparsity={sparsity:.1f}%")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.sparsity_level()
    print(f"\n  Final test accuracy : {test_acc:.2f}%")
    print(f"  Final sparsity level: {sparsity:.2f}%")
    return model, test_acc, sparsity


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(
    model: SelfPruningNet,
    lambda_val: float,
    save_path: str = "gate_distribution.png",
) -> None:
    """
    Histogram of all gate values in the final model.
    A successful result shows:
      - A large spike at 0  (pruned weights)
      - A cluster of values away from 0 (active weights)
    """
    gates = model.all_gate_values()
    pruned_pct = 100.0 * (gates < 0.1).sum() / len(gates)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(gates, bins=120, color="steelblue", edgecolor="white", linewidth=0.2)
    ax.set_xlabel("Gate Value  [sigmoid(gate_score)]", fontsize=12)
    ax.set_ylabel("Number of weights", fontsize=12)
    ax.set_title(
        f"Gate Value Distribution  (λ = {lambda_val})\n"
        f"{pruned_pct:.1f}% of weights pruned (gate < 0.1)",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    EPOCHS     = 10
    BATCH_SIZE = 256
    # Three λ values to demonstrate the sparsity-vs-accuracy trade-off
    LAMBDAS    = [1e-4, 1e-3, 1e-2]   # low, medium, high

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results: list[tuple[float, float, float]] = []
    best_model, best_lambda = None, LAMBDAS[1]

    for lam in LAMBDAS:
        model, test_acc, sparsity = run_experiment(
            lam, EPOCHS, device, train_loader, test_loader)
        results.append((lam, test_acc, sparsity))

        # Track the medium-λ model as our "best" for the plot
        if lam == LAMBDAS[1]:
            best_model, best_lambda = model, lam

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print(f"{'Lambda':<12}  {'Test Accuracy':>15}  {'Sparsity Level':>15}")
    print("-"*60)
    for lam, acc, sp in results:
        print(f"{lam:<12}  {acc:>14.2f}%  {sp:>14.2f}%")
    print("="*60)

    # ── Gate distribution plot ────────────────────────────────────────────────
    if best_model is not None:
        plot_gate_distribution(best_model, best_lambda)

    print("\nDone.")


if __name__ == "__main__":
    main()

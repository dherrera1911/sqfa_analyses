import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sqfa_matplotlib")

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "embeddings")
BATCH_SIZE = 256
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def output_path(split_name):
    return os.path.join(EMBEDDINGS_DIR, f"cifar100_resnet18_{split_name}.pt")


def extract_split(split_name, train):
    print(f"Extracting {split_name} embeddings")
    dataset = torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=WEIGHTS.transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    model = resnet18(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)

    x_all = []
    y_all = []
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            x_all.append(model(images).detach().cpu())
            y_all.append(labels.cpu())
            if batch_idx % 25 == 0:
                print(
                    f"{split_name}: {min((batch_idx + 1) * BATCH_SIZE, len(dataset))} "
                    f"of {len(dataset)}"
                )

    torch.save(
        {
            "X": torch.cat(x_all, dim=0).to(dtype=torch.float32),
            "y": torch.cat(y_all, dim=0).to(dtype=torch.long),
        },
        output_path(split_name),
    )
    print(f"Saved {output_path(split_name)}")


extract_split("train", train=True)
extract_split("test", train=False)

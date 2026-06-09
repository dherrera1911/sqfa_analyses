import os

import torch
from sklearn.decomposition import PCA


EMBEDDINGS_DIR = "embeddings"
INPUT_STEM = "cifar100_resnet18"
OUTPUT_STEM = "cifar100_resnet18_pca128"
N_COMPONENTS = 128


train_saved = torch.load(
    os.path.join(EMBEDDINGS_DIR, f"{INPUT_STEM}_train.pt"),
    map_location="cpu",
)
test_saved = torch.load(
    os.path.join(EMBEDDINGS_DIR, f"{INPUT_STEM}_test.pt"),
    map_location="cpu",
)

x_train = torch.as_tensor(train_saved["X"], dtype=torch.float32).numpy()
y_train = torch.as_tensor(train_saved["y"], dtype=torch.long)
x_test = torch.as_tensor(test_saved["X"], dtype=torch.float32).numpy()
y_test = torch.as_tensor(test_saved["y"], dtype=torch.long)

pca = PCA(n_components=N_COMPONENTS)
pca.fit(x_train)

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

torch.save(
    {
        "X": torch.as_tensor(x_train_pca, dtype=torch.float32),
        "y": y_train,
    },
    os.path.join(EMBEDDINGS_DIR, f"{OUTPUT_STEM}_train.pt"),
)
torch.save(
    {
        "X": torch.as_tensor(x_test_pca, dtype=torch.float32),
        "y": y_test,
    },
    os.path.join(EMBEDDINGS_DIR, f"{OUTPUT_STEM}_test.pt"),
)

print(f"Saved {OUTPUT_STEM}_train.pt and {OUTPUT_STEM}_test.pt")

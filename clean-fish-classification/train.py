import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import FishClassifier
from dataset import FishDataset
from transforms import build_transforms

# baseline config
ROOT_DIR = "/Users/rhyslittler/clean-fish-classification/fish_image"
BATCH_SIZE = 32
NUM_CLASSES = 23
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
IMG_SIZE = 128

# usual cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# build the transforms that we defined in transforms.py
train_transforms00000, val_transforms00000 = build_transforms(img_size=IMG_SIZE)

# create datasets
train_dataset = FishDataset(root_dir=ROOT_DIR, transform=train_transforms00000)
# If you have a separate validation set, create it here:
# val_dataset = FishDataset(root_dir=VAL_ROOT_DIR, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # remember to change this to 0 when using loal testing, when running on cluster will change to 4...
    pin_memory=True if torch.cuda.is_available() else False
)

# Create model
model = FishClassifier(num_classes=NUM_CLASSES)
# kinda cool so we can see the model summary; this is 237 for B0 layers in total...
# multiple convulational blocks
# batch normalization layers 
# activation layers
# depthwise/pointwise convolutions
# squeeze and excitation layers
print(model)
model = model.to(device)

# freeze the backbone, only train classifier head for now, after testing will change to two-stage approach traaining. 
for param in model.backbone.parameters():
    param.requires_grad = False

# validate that the classifier head is trainable
for param in model.backbone.classifier.parameters():
    param.requires_grad = True

#create cross entropy loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# kickoff the training loop
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"Training samples: {len(train_dataset)}")

for epoch in range(NUM_EPOCHS):
    model.train()
    
    training_loss = 0.0
    correct_predictions = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    # DataLoader returns (images, labels, tracking_ids) from the dataset, will think later how to add to this to bring multiple datasets together. 
    

    # wrapped the train loader in the progress bar, so we can call the progress bar as a generator for the trainloader data
    # so progress bar takes the train_loader (the data: images, labels, tracking_ids) and 
    # returns it as a generator, so we can call the progress bar as a generator for the trainloader data
    # took me a few mins lol 


    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass like usual to zero the gradients and get the outputs
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward pass to update the weights
        loss.backward()
        optimizer.step()

        # each iteration have to remember to update the metrics
        training_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total += labels.size(0)

        # now need to calculate the running averages to pass into bar
        running_loss = training_loss / total
        running_accuracy = correct_predictions / total

        # now pass the running averages into the progress bar for display
        # prints the number with 4 decimal places
        
        progress_bar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "accuracy": f"{running_accuracy:.4f}",
            "epoch": f"{epoch+1}/{NUM_EPOCHS}",
        })

    # calculate epoch metrics
    train_loss = training_loss / len(train_loader.dataset)
    train_accuracy = correct_predictions / total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

print("Training complete!")

torch.save(model.state_dict(), "fish_classifier.pth")
print("Model saved to fish_classifier.pth")

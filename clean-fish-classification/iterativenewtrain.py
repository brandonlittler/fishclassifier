import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import FishClassifier
from dataset import FishDataset
from transforms import build_transforms


NEW_DATA_DIR = "/Users/rhyslittler/clean-fish-classification/fish_image"  
MODEL_PATH = "fish_classifier.pth" 
BATCH_SIZE = 32
NUM_CLASSES = 23
LEARNING_RATE = 1e-4  
NUM_EPOCHS = 10
IMG_SIZE = 128

# Need to be able to train on the new folder, have to specify it for now...  
# then we need to work out which number of classes it is, so we can set the NUM_CLASSES variable. 
# splits to fish && 24, then we take the 1st index which is the 2nd part == 24
# so then int(24) == 24, this is the number of classes. 
NEW_FISH_FOLDER = "fish_25"
NUM_CLASSES = int(NEW_FISH_FOLDER.split("_")[1])



# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transforms, _ = build_transforms(img_size=IMG_SIZE)


new_dataset = FishDataset(root_dir=NEW_DATA_DIR, transform=train_transforms)
new_loader = DataLoader(
    new_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)




# Because we cannot resize the model, we need to load the old model and then expand the classifier head to the new number of classes. 
# so we create a new layer with the new number of classes and then load the old weights into the model. 
# Create model with new number of classes 
# This will create a brand new model with:
# backbone, randomly initiated.. 
# classifier head, randomly initiated..  = number of outputs as specified 
model = FishClassifier(num_classes=NUM_CLASSES)

# Load old weights (strict=False allows mismatched classifier size)


# Had to include this try block despite strict = false
#  The fix will remove mismatched classifier weights before loading... 
#  1. Create a new model with 24 classes 
#  2. Load the backbone weights 
#  3 Skip the classifier weights (they dont match as old vs new )
#  4 The new classifier will start random and learn during training, instead of being initialized with the old weights. 
try:
    old_state = torch.load(MODEL_PATH, map_location=device)
    # Handle class expansion properly - preserve old weights for existing classes
    if 'backbone.classifier.1.weight' in old_state:
        old_num_classes = old_state['backbone.classifier.1.weight'].shape[0]
        if old_num_classes != NUM_CLASSES:
            print(f"⚠️  Expanding model from {old_num_classes} to {NUM_CLASSES} classes")
            
            # Load the model first to get the new classifier structure
            # Temporarily remove classifier weights to load backbone
            temp_state = old_state.copy()
            old_classifier_weight = temp_state.pop('backbone.classifier.1.weight')
            old_classifier_bias = temp_state.pop('backbone.classifier.1.bias')
            
            # Load backbone weights
            model.load_state_dict(temp_state, strict=False)
            
            # Now copy old classifier weights to new classifier (preserving old classes)
            with torch.no_grad():
                new_weight = model.backbone.classifier[1].weight.data
                new_bias = model.backbone.classifier[1].bias.data
                
                # Copy old weights for existing classes (0 to old_num_classes-1)
                new_weight[:old_num_classes] = old_classifier_weight
                new_bias[:old_num_classes] = old_classifier_bias
                
                # New class(es) remain randomly initialized (this is fine)
                print(f"   ✅ Preserved classifier weights for classes 0-{old_num_classes-1}")
                print(f"   ✅ New class(es) {old_num_classes}-{NUM_CLASSES-1} randomly initialized")
        else:
            # Same number of classes, load normally
            model.load_state_dict(old_state, strict=False)
            print(f"✅ Loaded existing model with {NUM_CLASSES} classes")
    else:
        model.load_state_dict(old_state, strict=False)
        print(f"✅ Loaded existing model")
except FileNotFoundError:
    print(f"⚠️  No existing model found - starting from scratch")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    print(f"   Starting from scratch")

model = model.to(device)








# Freeze backbone, only train classifier
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.classifier.parameters():
    param.requires_grad = True

# Setup loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# Training loop
print(f"\nStarting training on new data for {NUM_EPOCHS} epochs...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    model.train()
    
    training_loss = 0.0
    correct_predictions = 0
    total = 0

    progress_bar = tqdm(new_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total += labels.size(0)

        running_loss = training_loss / total
        running_accuracy = correct_predictions / total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "loss_avg": f"{running_loss:.4f}",
            "acc": f"{running_accuracy:.4f}",
        })

    train_loss = training_loss / len(new_loader.dataset)
    train_accuracy = correct_predictions / total

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print("-" * 60)

# Save updated model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n Model updated and saved to {MODEL_PATH}")
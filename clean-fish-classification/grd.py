import gradio as gr
import torch
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

# import the model and transforms from the model.py and transforms.py files
from model import FishClassifier
from transforms import build_transforms

# load the labels from the json file, so we can map class to real fish name. 
with open("labels.json") as f:
    labels = json.load(f)

# set the device to cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the model with the number of classes in the labels file
model = FishClassifier(num_classes=len(labels))

# load the model weights from the fish_classifier.pth file
# Handle class expansion properly - remove classifier weights if size doesn't match
try:
    old_state = torch.load("fish_classifier.pth", map_location=device)
    # Check if classifier size matches
    if 'backbone.classifier.1.weight' in old_state:
        old_num_classes = old_state['backbone.classifier.1.weight'].shape[0]
        if old_num_classes != len(labels):
            # Remove classifier weights if size doesn't match (for class expansion)
            del old_state['backbone.classifier.1.weight']
            del old_state['backbone.classifier.1.bias']
            print(f"⚠️  Model has {old_num_classes} classes, but labels has {len(labels)}. Classifier will be randomly initialized.")
    
    model.load_state_dict(old_state, strict=False)
    print(f"✅ Model loaded successfully with {len(labels)} classes")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    print(f"   Model will use randomly initialized weights")

# set the model to evaluation mode
model.eval().to(device)

# this will call the transform builder
# returns the train and validation transforms
# apply the transforms and it is expecting 128x128 inputs. 
_, val_transforms = build_transforms(img_size=128)

# Initialize global variable for storing current prediction
current_fish_name = ""


# define the predict_fish function, this is the function that will be called when the user uploads an image.
# GRADIO will pass in a PIL image object, so we need to convert it to a tensor.
def predict_fish(image):
    global current_fish_name
    if image is None:
        return "Upload an image of fish for classification. It will return the top 3 most likely fish species.", "", ""
    
    # Validate that we have a PIL Image object
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            return "Error: Expected PIL Image object. Please upload an image through the UI.", "0%", "Invalid image type"
    except ImportError:
        pass  
    
    try:
        # convert the image to a tensor
        # convert the image to RGB
        # unsqueeze the tensor to add a batch dimension
        # move the tensor to the device
        img_tensor = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
        
        # set the model to evaluation mode
        # this is because we are not training the model, we are just using it to classify the image that we uploaded via UI. 
        # torch.no_grad() is used to disable gradient calculation
        with torch.no_grad():

            # this is a forward pass through the model, we are passing the tensor through the model to get the outputs.
            # the output shape is going to be (1, x number of classes)
            # these are the logits, we need to convert them to probabilities.
            
            # reason the output is 1, x is because it's 1 image that we are passing 
            # and the X will be the number of classes in the labels file with the probability scores against each
            # then further we will need to convert the probabilities to a class index and a confidence score
            outputs = model(img_tensor)

            # converts the logits to probabilities
            # dim=1 means that the probabilities are calculated along the number of classes dimension which is the 2nd dimension 
            # we starteed with 23, but each time we retrain then this is going to increase accordingly
            probs = torch.softmax(outputs, dim=1)



            # finds which index is the highest probability
            # .item() is used to get the value of the tensor
            # we are getting the index of the highest probability
            pred_class = outputs.argmax(dim=1).item()
            # gets the confidence score for the highest probability
            confidence = probs[0][pred_class].item()
            # top-k: use at most 3, or fewer if we have fewer classes
            num_classes = probs.shape[1]
            k = min(3, num_classes)
            topk_probs, topk_indices = torch.topk(probs, k)
            
            # Debug: Print top predictions to help diagnose issues
            print(f"DEBUG: Top prediction: class {pred_class} ({labels[str(pred_class)]}) with confidence {confidence:.2%}", flush=True)
            print(f"DEBUG: Top {k} predictions:", flush=True)
            for i in range(k):
                idx = topk_indices[0][i].item()
                prob = topk_probs[0][i].item()
                print(f"  {i+1}. Class {idx} ({labels[str(idx)]}): {prob:.2%}", flush=True)
        
        # look up the fish name from the labels file
        fish_name = labels[str(pred_class)]
        current_fish_name = fish_name
        # format the top-k predictions for UI (k may be 1, 2, or 3)
        top3_text = "\n".join([f"{i+1}. {labels[str(topk_indices[0][i].item())]}: {topk_probs[0][i].item():.2%}" for i in range(k)])
        
        return fish_name, f"{confidence:.2%}", top3_text
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(f"Prediction error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return error_msg, "0%", "Error occurred - check logs"

# two functions for correct and incorrect buttons in the UI/ feedback, log into a CSV file. 
# this will help us to improve the model over time

def mark_correct():
    global current_fish_name
    if not current_fish_name:
        return "Please make a prediction first!"
    with open("feedback.csv", "a") as f:
        f.write(f"{current_fish_name},correct\n")
    return f"You have CONFIRMED this fish as being the following species: {current_fish_name}"

def mark_incorrect():
    global current_fish_name
    if not current_fish_name:
        return "Please make a prediction first!"
    with open("feedback.csv", "a") as f:
        f.write(f"{current_fish_name},incorrect\n")
    return f"You have confirmed this fish as NOT being the following species: {current_fish_name}"


# Modern X-like theme with clean, minimal design
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate"
).set(
    body_background_fill="*neutral_50",
    body_text_color="*neutral_900",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_secondary_background_fill="*neutral_200",
    button_secondary_background_fill_hover="*neutral_300",
    border_color_primary="*neutral_200",
    border_color_accent="*primary_300",
    shadow_spread="0",
    shadow_drop="0 1px 2px 0 rgba(0, 0, 0, 0.05)"
)

custom_css = """
    .gradio-container {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    .markdown {
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    .markdown h1 {
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }
    .markdown p {
        color: #536471 !important;
        margin-bottom: 16px !important;
    }
    .image-container {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .textbox {
        border-radius: 8px !important;
    }
    .button {
        border-radius: 24px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    """

with gr.Blocks(title="BRL Scuba Classifier") as fishclassifier:
    gr.Markdown("### BRL Scuba Classifier")
    gr.Markdown(f"Upload an image to identify the fish species. Trained on 32+ species.")

    image_input_box = gr.Image(type="pil", label="Upload Image", height=400)

    with gr.Row():
        species_output_box = gr.Textbox(label="Species", interactive=False, scale=2)
        confidence_output_box = gr.Textbox(label="Confidence", interactive=False, scale=1)
    
    top3_output_box = gr.Textbox(label="Top 3 Predictions", interactive=False, lines=3)

# now for the two buttons side by side:
    with gr.Row():
        correct_button = gr.Button("✓ Correct", variant="primary", scale=1)
        incorrect_button = gr.Button("✗ Incorrect", variant="secondary", scale=1)

    feedback_message = gr.Textbox(label="Feedback", interactive=False)

# when the image is uploaded we need to run the predict_fish function that we have defined earlier
# this is called from the prediction model that we have defined in the other classes 
    image_input_box.change(
            fn=predict_fish,
            inputs=image_input_box,
            outputs=[species_output_box, confidence_output_box, top3_output_box]
        )
    
        # Connect the buttons
    correct_button.click(fn=mark_correct, outputs=feedback_message)
    incorrect_button.click(fn=mark_incorrect, outputs=feedback_message)



fishclassifier.launch(theme=custom_theme, css=custom_css)
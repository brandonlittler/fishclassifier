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
#load_state_dict is used to load the model weights from the fish_classifier.pth file
# strict = false means that the model will load the weights even if the number of classes is not the same as the number of classes in the labels file
model.load_state_dict(torch.load("fish_classifier.pth", map_location=device), strict=False)

# set the model to evaluation mode
model.eval().to(device)

# this will call the transform builder
# returns the train and validation transforms
# apply the transforms and it is expecting 128x128 inputs. 
_, val_transforms = build_transforms(img_size=128)

# define the predict_fish function, this is the function that will be called when the user uploads an image.
# GRADIO will pass in a PIL image object, so we need to convert it to a tensor.
def predict_fish(image):
    global current_fish_name
    if image is None:
        return "Upload an image of fish for classification. It will return the top 3 most likely fish species.", "", ""
    
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
        # finds the top 3 probabilities and their indices
        # this is used to get the top 3 most likely fish species
        top3_probs, top3_indices = torch.topk(probs, 3)
    
    # look up the fish name from the labels file
    fish_name = labels[str(pred_class)]
    current_fish_name = fish_name
    # then just concat and format the top 3 most likely fish species for UI visualisation
    top3_text = "\n".join([f"{i+1}. {labels[str(top3_indices[0][i].item())]}: {top3_probs[0][i].item():.2%}" for i in range(3)])
    
    return fish_name, f"{confidence:.2%}", top3_text

# two functions for correct and incorrect buttons in the UI/ feedback, log into a CSV file. 
# this will help us to improve the model over time

def mark_correct():
    global current_fish_name
    return f"You have CONFIRMED this fish as being the following species: {current_fish_name}"
    with open("feedback.csv", "a") as f:
        f.write(f"{fish_name},correct\n")
   

def mark_incorrect():
    global current_fish_name
    return f"You have confirmed this fish as NOT being the following species: {current_fish_name}"
    with open("feedback.csv", "a") as f:
        f.write(f"{fish_name},incorrect\n")


with gr.Blocks(title = "Rhys' Scuba Diving Fish Classifier") as fishclassifier:
    gr.Markdown("Rhys' Scuba Diving Fish Classifier")
    gr.Markdown(f"Upload an image of a fish to classify it into one of the {len(labels)} trained species.")

    image_input_box = gr.Image(type="pil", label="Upload A JPEG or PNG Image of a Fish")

    species_output_box = gr.Textbox(label="Species", interactive=False)
    confidence_output_box = gr.Textbox(label="Confidence", interactive=False)
    top3_output_box = gr.Textbox(label="Top 3", interactive=False)

# now for the two buttons side by side:
    with gr.Row():
        correct_button = gr.Button("Correct", variant="primary")
        incorrect_button = gr.Button("Incorrect", variant="stop")

    feedback_message = gr.Textbox(label="Prediction Feedback", interactive=False)

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



fishclassifier.launch()
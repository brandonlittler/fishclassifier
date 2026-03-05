# In this class, will define how to access one sampleset of data 
#  No splitting, dataLoaders, or training loop. 
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


# This is going to re-index the folder names to be 0-indexed, PyTorch needs this... 
# fish_01 -> 0 etc... 
def folder_to_label(folder_name: str) -> int:
   
   # Then return the index of the folder name
   
    return int(folder_name.split("_")[1]) - 1

    # folder name split will turn fish_01 -> ["fish", "01"]
    # [1] grabs the 01 
    # int(01) can then become 1 
    # -1 then to get to 0 indexed, as we need this for PyTorch. 


# This will extract the tracking ID from the filename, this is the unique identifer of the fish, not the class of fish. 
def filename_to_tracking_id(filename: str) -> str:
    # "fish_000000009598_05281.png" -> "000000009598"
    return Path(filename).stem.split("_")[1]

    #.stem will get the filename without the extension
    # split("_") will turn "fish_000000009598_05281.png" -> ["fish", "000000009598", "05281.png"]
    # [1] grabs the 000000009598
    # return this as a string
    # then we have the complete identifer of the fish in a string format, later we can use this to track the fish across frames. 


# Now, we actually have to define the dataset that PyTorch will be able to use later in the dataLoader. 

"""
This is going to return x, label_idx, tracking_id
x is the image data, as a tensor, this is like our passport photo, will be stored as pixel values /// (w/transforms applied)
label_idx is the index of the class of fish, as an integer, this is like our nationality in passport /// (0-22)
tracking_id is the unique identifier of the fish, as a string, this is like our passport ID /// (000000009598....)
"""
class FishDataset(Dataset):

    # root_dir is where the dataset will be, for this purpose probably just use local file repo 
    # transform will be train or val transforms applied to the image data 
    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None):
        

        # Convert the root_dir to a Path object, this is so we can use the Path object methods later. 
        self.root_dir = Path(root_dir)
        # store transform function that will be applied to the image in getitem later.
        self.transform = transform

        # Need to create an empty list to store the samples of tuples, including the image path, label index, and tracking id. 
        self.samples = []  

        # Loop through each class folder of fish, this is going to be the 23 different classes of fish for this example.... 
        # for example, look at each fish folder fish_01, fish_02, fish_03, etc... 
        # sorted just means it will go in order, 1,2,3....

        for class_dir in sorted(self.root_dir.glob("fish_*")):
            # ignore anything that is not a folder, and flag it as an error, need to figure out later how to catch this in UI. 
            if not class_dir.is_dir():
                continue

            # take the folder name of fish_07 for example and give it a label index of 6, as we are 0 indexed. 
            label_idx = folder_to_label(class_dir.name)
    

     # Loop and find all the images in the class folder .png only
            for img_path in sorted(class_dir.glob("*.png")):
                 # take just the fish filename and turn it into a tracking id (passport ID)
                tracking_id = filename_to_tracking_id(img_path.name)
                 # add a row to the dataset, including the 3 values we need for training. 
                self.samples.append((img_path, label_idx, tracking_id))

        # If no samples are found, raise an error, figure out how to catch this/act on it later in UI. 
        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root_dir}")

     # This will return the number of samples in the dataset so that the DataLoader can know how many to sample later on... 
    def __len__(self) -> int:
        return len(self.samples)

    # This will return the image data, label index, and tracking id for the given index when called by the DataLoader. 
    def __getitem__(self, idx: int):
         # pull the image path, label index, and tracking id for the given index from the dataset we created in init, this was called samples....
        img_path, label_idx, tracking_id = self.samples[idx]
         # open the image and convert it to RGB values, here we are loading it from the file path into memory
        img = Image.open(img_path).convert("RGB")
         # apply the transform function to the image, if there are any and RETURN the TENSOR
        x = self.transform(img) if self.transform else img
        # return the tensor, label index, and tracking id
        return x, label_idx, tracking_id
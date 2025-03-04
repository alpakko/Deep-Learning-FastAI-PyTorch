# *** - Disabling CUDA, MPS and GPU detection, forcing CPU usage - im using old intel Mac 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ***

from fastai.vision.all import *
import torch

device = torch.device('cpu') # ***
torch.cuda.is_available = lambda: False  # ***
torch.backends.mps.is_available = lambda: False  # ***

data_dir = untar_data(URLs.PETS, c_key='data') 
path = data_dir/'images' 

if not path.exists():
    path = untar_data(URLs.PETS, c_key='data', exist_ok=True)/'images'
else:
    print("Dataset already downloaded, using local copy.")

def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224), device=device)

learn = vision_learner(dls, resnet34, metrics=error_rate, pretrained=True, normalize=False)

if True:  # Change to False if you want to skip normalization entirely
    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    norm_tfms = Normalize.from_stats(*stats, cuda=False)  # ***
    dls.add_tfms([norm_tfms], 'after_batch')

learn.dls.device = device # ***
learn.model.to(device) # ***

learn.fine_tune(1)

learn.save('pets_model')  
print(f"Model saved to {path}/models/pets_model.pth")

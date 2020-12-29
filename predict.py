import argparse
import json
import torch
import numpy as np

from PIL import Image
from torchvision import datasets, transforms, models


def parserFunc():
    parser = argparse.ArgumentParser(description = "Prediction environment setup")
    
    parser.add_argument('--top_k', type=int, default=5, help='Set the value of topk')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Cateogry to predict')
    parser.add_argument('--gpu', type=str, default='gpu', help='For using GPU')
    parser.add_argument('--checkpoint', type=str,default="checkpoint.pth",help='File for loading checkpoint')
    parser.add_argument('--image_path', type=str, help='Path of image')
    
    args = parser.parse_args()
    return args

def define_model(arch):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = arch
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model



def load_checkpoint(filepath):
    load_dict = torch.load(filepath)
    
    model = define_model(load_dict['architecture'])
    model.classifier = load_dict['classifier']
    model.load_state_dict(load_dict['state_dict'])
    model.class_to_idx = load_dict['class_to_idx']
    
    
      
    return model


def process_image(image):
    
    pil_image = Image.open(image)

    
    modifyImage = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    img = modifyImage(pil_image)
    
    return img
    

def image_show(image, ax=None, title=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    

    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image, model, top_k, device,category_file):
    temp = model.eval()
    
    model.to('cpu')

    image = torch.from_numpy(np.expand_dims(image,axis=0)).type(torch.FloatTensor)
    
    prob = torch.exp(model.forward(image))
    
    
    top_prob, top_label = prob.topk(top_k)
       
    top_prob = np.array(top_prob.detach())[0] 
    
    top_label = np.array(top_label.detach())[0]
    
    idx_to_class = {val: key for key,val in model.class_to_idx.items()}
    
    top_label = [idx_to_class[label] for label in top_label]
   
    
    return top_prob, top_label
    

def main():
    arguments = parserFunc()
    
    model = load_checkpoint(arguments.checkpoint)
    
    with open(arguments.category_names, 'r') as f:
        category_file = json.load(f)
        
    modify_image = process_image(arguments.image_path)
    
    if not arguments.gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    
    topk_prob, topk_label= predict(modify_image, model, arguments.top_k, device, category_file)
    
    print("Top labels = ", topk_label)
    
    print("Top Probability = ", topk_prob)


if __name__ == '__main__':
    main()
    
import argparse

import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


arch_units = {"vgg16":25088, "densenet121":1024, "alexnet":9216}


def parserFunc():
    parser = argparse.ArgumentParser(description = "Image classifier setup")
    
    parser.add_argument('--save_dir', type=str, action='store', default='checkpoint.pth',help='Save checkpoint directory')
    parser.add_argument('--arch',type=str, default='vgg16', help='Choose torchvision.models architecture (Possible archsa are denset, alexnet and vgg16). Default is vgg16')
    parser.add_argument('--learning_rate',type=float,  default=0.001, help='Define learning rate')
    parser.add_argument('--hidden_units',type=int,default=4096,help='Define hidden unit size')
    parser.add_argument('--epochs',type=int,default=5,help='Define number of epochs')
    parser.add_argument('--gpu',type=bool,dest='gpu',default='gpu',help='For using GPU')
    
    args = parser.parse_args()
    return args

def trainData(train_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle= True)
    
    return train_data, trainloader

    
def testData(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return test_data, testloader
    
def validData(valid_dir):
    
    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return valid_data, validloader
    
    
def define_model(arch):
    
    model = eval("models.{}(pretrained=True)".format(arch))
    model.name = arch
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model



def define_classifier(arch, hidden_units):
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(arch_units[arch.name], hidden_units, bias=True)), ('relu1', nn.ReLU()),('dropout1', nn.Dropout(p=0.5)), ('fc2', nn.Linear(hidden_units, 102, bias=True)), ('output', nn.LogSoftmax(dim=1))]))
    
    return classifier



def training_process(epochs, model, trload, teload, criterion, optimizer, device):
    print_every = 30
    steps = 0
    
    
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs,labels in iter(trload):
            steps +=1
            
            inputs,labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            loss_ps = model.forward(inputs)
            loss = criterion(loss_ps,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                
                with torch.no_grad():
                    valid_loss = 0
                    acc = 0
                    
                    for inp, lab in iter(teload):
                        inp,lab = inp.to(device), lab.to(device)
                        
                        out = model.forward(inp)
                        valid_loss += criterion(out, lab).item()
                        
                        ps = torch.exp(out)
                        eq = (lab.data == ps.max(dim=1)[1])
                        acc += eq.type(torch.FloatTensor).mean()
                        
                        
                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(teload)),
                     "Validation Accuracy: {:.4f}".format(acc/len(teload)))  
                
                running_loss = 0
                model.train()
                
      
    return model              

    
def save_checkpoint(train_model, directory ,trdata):
    
    
    train_model.class_to_idx = trdata.class_to_idx
        
    checkpoint = {'architecture': train_model.name,
                  'state_dict': train_model.state_dict(),
                  'class_to_idx': train_model.class_to_idx,
                  'classifier': train_model.classifier,
                  }
     
    torch.save(checkpoint, directory)
    
    
def main():
    
    print("3 models available : vgg16, alexnet and densenet121.")
    
    arguments = parserFunc()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trdata, trload = trainData(train_dir)
    tesdata, teload = testData(test_dir)
    valdata, valload = validData(valid_dir)
    
    model = define_model(arch = arguments.arch)
    
    model.classifier = define_classifier(model, arguments.hidden_units)
    
    if not arguments.gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    
    
    learning_rate = arguments.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
        
    train_model = training_process(arguments.epochs, model, trload, teload, criterion, optimizer, device)
    
     
    if arguments.save_dir:
        save_dir = arguments.save_dir
    else:
        save_dir = 'checkpoint.pth'
        
    save_checkpoint(train_model, save_dir,trdata)
    
    

if __name__ == '__main__':
    main()



    


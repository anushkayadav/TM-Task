import io

import torch
import torch.nn as nn
import torchvision.models as models
#from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2 
from torchvision import datasets

def get_model():
	use_cuda = torch.cuda.is_available()
	checkpoint_path = 'model_transfer.pt'
	res50 = models.resnet50(pretrained=True)
	#model = models. densenet161(pretrained=True)
	#model.classifier = nn.Linear(2208, 102)
	model_transfer=res50

	for name,child in model_transfer.named_children():
		if name in ['fc']:
			for param in child.parameters():
				param.requires_grad = True
		else:
			for param in child.parameters():
				param.requires_grad = False

	model_transfer.fc = nn.Sequential(nn.Linear(2048, 516),nn.ReLU(inplace=True),nn.Linear(516,64),nn.ReLU(inplace=True),nn.Linear(64,3))
	if use_cuda:
		model_transfer = model_transfer.cuda()

	checkpoint = torch.load('model_transfer.pt')
	model_transfer.load_state_dict(checkpoint['state_dict'])
	model_transfer.eval()
	return model_transfer



def get_tensor(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(256),
        				    transforms.CenterCrop(224),
        				    transforms.ToTensor(),
        				    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             					  std=[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
	
	return my_transforms(image).unsqueeze(0)
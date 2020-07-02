import json
import torch
from commons import get_model, get_tensor
import os
from torchvision import datasets
import torchvision.transforms as transforms

train_dir = 'images/train'


train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


#train_data = datasets.ImageFolder(train_dir , transform=train_transforms)
'''with open('cat_to_name.json') as f:
	cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
	class_to_idx = json.load(f)

idx_to_class = {v:k for k, v in class_to_idx.items()}'''

model = get_model()

class_names =['kurti', 'saree', 'shirt']

#class_names = [item[4:].replace("_", " ") for item in train_data.classes]

#fp=train_data.classes

def get_flower_name(image_bytes):
	tensor = get_tensor(image_bytes)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	with torch.no_grad():
		out = model(tensor.to(device))
		ps = torch.exp(out)
		top_p, top_class = ps.topk(1, dim=1)
		index = top_class.item()

	return class_names[index]



	'''outputs = model.forward(tensor)
	_, prediction = outputs.max(1)
	category = prediction.item()
	class_idx = idx_to_class[category]
	flower_name = cat_to_name[class_idx]
	return category, flower_name'''
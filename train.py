import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
#import test
#import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

##REPRODUCIBILITY
torch.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = '../seg_train'

result_train_loss=[]
result_train_acc=[]
x1=range(50)

result_test_loss=[]
result_test_acc=[]
x2=range(0,50,10)

def adjust_learning_rate(optimizer, epoch):
	lr=0.01*(0.1**(epoch//30))
	for param_group in optimizer.param_groups:
		param_group['lr']=lr

def train():
	data_transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	#print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
	#print(train_set.num_classes)
	model = VGG16(num_classes=train_set.num_classes)
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	count = 100.0
	num_epochs = 35
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

	for epoch in range(num_epochs):
		print(f'Epoch: {epoch + 1}/{num_epochs}')
		print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))
		adjust_learning_rate(optimizer,num_epochs);
		training_loss = 0.0
		training_corrects = 0

		for i, (inputs, labels) in enumerate(data_loader):
			inputs = Variable(inputs.cuda(CUDA_DEVICES))
			labels = Variable(labels.cuda(CUDA_DEVICES))			

			optimizer.zero_grad()

			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc =training_corrects.double() /len(train_set)

		result_train_loss.append(str(training_loss))     #train_data
		result_train_acc.append(str(training_acc))

		if (epoch+1)%10==0:
			#torch.save(model, f'model.pth')
			#tmp1,tmp2=test.test(f'model.pth')
			#print(str(tmp1)+'\t'+str(tmp2))
			torch.save(model, f'model-{count:.02f}-train.pth')
			#result_test_loss.append(str(tmp1))
			#result_test_acc.append(str(tmp2))
			count=count+1

		#print(training_acc.type())
		#print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model_params)
	torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')


if __name__ == '__main__':
	train()
	train_loss_file=open('train_loss_file.txt', 'w')
	train_acc_file=open('train_acc_file.txt', 'w')
	#test_loss_file=open('test_loss_file.txt', 'w')
	#test_acc_file=open('test_acc_file.txt', 'w')
	
	for i in result_train_loss:
		train_loss_file.write(i)
		train_loss_file.write('\n')
	for i in result_train_acc:
		train_acc_file.write(i)
		train_acc_file.write('\n')
	#for i in result_test_loss:
	#	test_loss_file.write(i)
	#	test_loss_file.write('\n')
	#for i in result_test_acc:
	#	test_acc_file.write(i)
	#	test_acc_file.write('\n')
	train_loss_file.close()
	train_acc_file.close()
	#test_loss_file.close()
	#test_acc_file.close()


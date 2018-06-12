import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata

n_level = 3
n_epoch = 100
n_it = 10
train_batch = 128
test_batch = 12
A = 10
B = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_0(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

test_dataset = tdata.TensorDataset(images_test, labels_test)
test_loader = tdata.DataLoader(test_dataset, batch_size=test_batch)

net[0] = CNN_0()
net[1] = CNN_1()
net[2] = CNN_2()

# criterion = nn.CrossEntropyLoss()
# for i range(n_level):
#     optimizer[i] = optim.SGD(net[i].parameters(), lr=0.001, momentum=0.9)
criterions = [nn.CrossEntropyLoss() for i in range(n_level)]
optimizers = [optim.SGD(net[i].parameters(), lr=0.001, momentum=0.9) for i in range(n_level)]

for ep in range(n_epoch):

    for le in range(n_level):
        images_t[le] = np.array(images_update[le])
        labels_t[le] = np.array(labels_update[le])

        images_t[le] = torch.tensor(images_t[le])
        labels_t[le] = torch.tensor(labels_t[le])

        train_dataset[le] = tdata.TensorDataset(images_t[le], labels_t[le])
        train_loader[le] = tdata.DataLoader(train_dataset[le], 
                        batch_size=train_batch, shuffle=True)
    
        running_loss = 0.0

        for it in range(n_it):
            if images_t.size()[0] < 1:
                break
            for i, data in enumerate(train_loader[le], 0):
                # get the inputs
                images, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net[le](images)
                loss[le] = criterion[le](outputs, labels)
                loss[le].backward()
                optimizer[le].step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (it + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        
        ### update
        with torch.no_grad():
            for i, data in enumerate(train_loader[le], 0):
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)

                images_update[le] = images_t[le][outputs.numpy()==predicted.numpy()]
                labels_update[le] = labels_t[le][outputs.numpy()==predicted.numpy()]

                images_tmp = images_t[le][outputs.numpy()!=predicted.numpy()]
                labels_tmp = labels_t[le][outputs.numpy()!=predicted.numpy()]

                if le < 1:
                    for j in range(images_tmp.shape[0]):
                        if xxx > xxx:
                            images_update[le+1] = np.vstack((images_update[le+1], np.expand_dims(images_tmp[j], axis=0)))
                            labels_update[le+1] = np.vstack((labels_update[le+1], np.expand_dims(labels_tmp[j], axis=0)))
                            # move to 1
                elif le < 2:
                    for j in range(images_tmp.shape[0]):
                        if xxx > xxx:
                            if xx > xx:
                                images_update[le-1] = np.vstack((images_update[le-1], np.expand_dims(images_tmp[j], axis=0)))
                                labels_update[le-1] = np.vstack((labels_update[le-1], np.expand_dims(labels_tmp[j], axis=0)))
                                # move to 0
                            else:
                                images_update[le+1] = np.vstack((images_update[le+1], np.expand_dims(images_tmp[j], axis=0)))
                                labels_update[le+1] = np.vstack((labels_update[le+1], np.expand_dims(labels_tmp[j], axis=0)))
                                # move to 2
                elif le < 3:
                    for j in range(images_tmp.shape[0]):
                        if xxx > xxx:
                            images_update[le-1] = np.vstack((images_update[le-1], np.expand_dims(images_tmp[j], axis=0)))
                            labels_update[le-1] = np.vstack((labels_update[le-1], np.expand_dims(labels_tmp[j], axis=0)))
                            # move to 2
                else:
                    print('out of level range!')
                    exit(0)
                
                # c = (predicted == labels).squeeze()
                # for j in range(test_batch):
                #     label = labels[j]
                #     class_correct[label] += c[j].item()
                #     class_total[label] += 1

                ###update according to predicted and labels

print('Finished Training')


 
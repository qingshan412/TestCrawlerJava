from __future__ import print_function, division
from io import open
import glob,random

import torch
import torch.nn as nn
import torch.optim as optim

### dict for data transform

RoadCode = {}
ExitCode = {}

def check_code(name, code):
    if name not in code:
        code[name] = len(code)
    return float(code[name])

def code_list(code):
    code_l = []
    for i in range(len(code)):
        code_l.append(0)
    for key,item in code.iteritems:
        code_l[item] = key
    return code_l

# Read a file and return all available instances in it
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    instances = []

    prev_id = None
    instance = [[],[],[],]
    
    for line in lines[1:]:
        line_list = line.strip().split(',')
        if line_list[0] != prev_id:
            if instance[2]:
                instances.append(instance)
            instance = [[],[],[],]
            for i in range(4):#plateType, vehicleType, PlateColor, weightLimit
                instance.append(float(line_list[i+2]))
            for i in range(4):#year, month, weekend71, hour
                instance[0].append(float(line_list[i+7]))
            instance[0].append(check_code(line_list[11], RoadCode))#KKRoadCode
            instance[0].append(float(line_list[12]))#KKLon
            instance[0].append(float(line_list[13]))#KKLat
            instance[0].append(check_code(line_list[14], RoadCode))#DestExit

            prev_id = line_list[0]
        else:
            if not instance[1]:
                for i in range(4):#year, month, weekend71, hour
                    instance[1].append(float(line_list[i+7]))
                instance[1].append(check_code(line_list[11], RoadCode))#KKRoadCode
                instance[1].append(float(line_list[12]))#KKLon
                instance[1].append(float(line_list[13]))#KKLat
                instance[1].append(check_code(line_list[14], RoadCode))#DestExit
            elif not instance[2]:
                for i in range(4):#year, month, weekend71, hour
                    instance[2].append(float(line_list[i+7]))
                instance[2].append(check_code(line_list[11], RoadCode))#KKRoadCode
                instance[2].append(float(line_list[12]))#KKLon
                instance[2].append(float(line_list[13]))#KKLat
                instance[2].append(check_code(line_list[14], RoadCode))#DestExit
            else:
                continue

    return instances

##############################################################################
#############    data preparation
##############################################################################


### parameters
path = './data/10/*.txt'

### check files
files = glob.glob(path)
print(files)

### get all available instances
### instance:[[first time-related features], [second], [third], static features, label]
inputs = []
for filename in files:
    lines = readLines(filename)
    inputs.extend(lines)
random.shuffle(inputs)
    
### make it convenient to find name with code
RoadCode_list = code_list(RoadCode)
ExitCode_list = code_list(ExitCode)

import torch.utils.data as data_utils

train = data_utils.TensorDataset(features, targets)
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

### divide time-related inputs, static inputs, and outputs
total_instances = len(inputs)
for i in total_instances:
    if i < 1:
        tr_input = torch.tensor(inputs[i][:3])
        st_input = torch.tensor(inputs[i][3:-1])
        output = torch.tensor(inputs[i][-1])
    else:
        tr_input = torch.stack((tr_input, torch.tensor(inputs[i][:3])))
        st_input = torch.stack((st_input, torch.tensor(inputs[i][3:-1])))
        output = torch.stack((output, torch.tensor(inputs[i][-1])))

### standardization
tr_input_sizes=tr_input.size()
tr_input_flat = tr_input.view(-1, tr_input_sizes[-1])
tr_input_mean = tr_input_flat.mean(dim=0, keepdim=True)
tr_input_mean = tr_input_mean.view(1,1,tr_input_sizes[-1])
tr_input_std = tr_input_flat.std(dim=0, keepdim=True)
tr_input_std = tr_input_std.view(1,1,tr_input_sizes[-1])

tr_input = (tr_input-tr_input_mean)/tr_input_std

st_input_mean = st_input.mean(dim=0, keepdim=True)
st_input_std = st_input.std(dim=0, keepdim=True)
st_input = (st_input-st_input_mean)/st_input_std

### format for training and test
num_train = int(0.8*total_instances)
tr_input_train, tr_input_test = tr_input.split([num_train, total_instances-num_train])
st_input_train, st_input_test = st_input.split([num_train, total_instances-num_train])
output_train, output_test = output.split([num_train, total_instances-num_train])

train = torch.utils.data.TensorDataset(tr_input_train, st_input_train, output_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = torch.utils.data.TensorDataset(tr_input_test, st_input_test, output_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=1)

##############################################################################
#############    model construction
##############################################################################

class RNN(nn.Module):
    def __init__(self, tr_input_size, st_input_size, output_size, hidden_size=3):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=tr_input_size, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size+st_input_size, output_size)


    def forward(self, tr_input, st_input):
        out_rnn, _ = self.rnn(tr_input, None)
        last_tensor = torch.mean(out_rnn, dim=1)
        out = self.fc(torch.cat((st_input, last_tensor),dim=1))
        
        return out



### initialize a model
tr_input_size = tr_input_sizes[-1]
st_input_size = st_input.size()[-1]
output_size = len(ExitCode_list)

ex_rnn = RNN(tr_input_size, st_input_size, output_size, hidden_size=2)


##############################################################################
#############    training process
##############################################################################

### GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device using: " + device)
# ex_rnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ex_rnn.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    runing_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        tr_input_i, st_input_i, output_i = data
        ### GPU
        # tr_input_i, st_input_i, output_i = tr_input_i.to(device), st_input_i.to(device), output_i.to(device)

        optimizer.zero_grad()

        output_i_r = ex_rnn(tr_input_i, st_input_i)
        loss = criterion(output_i_r, output_i)
        loss.back()
        optimizer.step

        running_loss += loss.item()

        if i%10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')


##############################################################################
#############    test process
##############################################################################

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        tr_input_i, st_input_i, out_put_i = data
        output_i_t = ex_rnn(tr_input_i, st_input_i)
        _, predicted = torch.max(output_i_t, 1)
        total += output_i.size(0)
        correct += (predicted == output_i).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (total, 
    100 * correct / total))

class_correct = list(0. for i in range(output_size))
class_total = list(0. for i in range(output_size))
with torch.no_grad():
    for data in testloader:
        tr_input_i, st_input_i, out_put_i = data
        output_i_t = ex_rnn(tr_input_i, st_input_i)
        _, predicted = torch.max(output_i_t, 1)
        c = (predicted == output_i).squeeze()
        class_correct[output_i[0]] += c[0].item()
        class_total[output_i[0]] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
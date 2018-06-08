from __future__ import print_function, division
from io import open
import glob,random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata

import numpy as np

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
    for key,item in code.iteritems():
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
            instance.append(check_code(line_list[14], ExitCode))#DestExit

            prev_id = line_list[0]
        else:
            if not instance[1]:
                for i in range(4):#year, month, weekend71, hour
                    instance[1].append(float(line_list[i+7]))
                instance[1].append(check_code(line_list[11], RoadCode))#KKRoadCode
                instance[1].append(float(line_list[12]))#KKLon
                instance[1].append(float(line_list[13]))#KKLat
                # instance[1].append(check_code(line_list[14], ExitCode))#DestExit
            elif not instance[2]:
                for i in range(4):#year, month, weekend71, hour
                    instance[2].append(float(line_list[i+7]))
                instance[2].append(check_code(line_list[11], RoadCode))#KKRoadCode
                instance[2].append(float(line_list[12]))#KKLon
                instance[2].append(float(line_list[13]))#KKLat
                # instance[2].append(check_code(line_list[14], ExitCode))#DestExit
            else:
                continue

    return instances

##############################################################################
#############    data preparation
##############################################################################
train_batch = 64
test_batch = 1
### parameters
path = './data/10/*.txt'

### check files
files = glob.glob(path)
print(files)

### get all available instances
### instance:[[first time-related features], [second], [third], static features, label]
inputs = []
for filename in files:
    print('process ' + filename + '...')
    lines = readLines(filename)
    inputs.extend(lines)

print(inputs[0])
random.shuffle(inputs)

### make it convenient to find name with code
RoadCode_list = code_list(RoadCode)
ExitCode_list = code_list(ExitCode)
print('RoadCode')
for i in range(len(RoadCode_list)):
    print(RoadCode_list[i]+':'+str(i))
# print('ExitCode')
# for i in range(len(ExitCode_list)):
#     print(ExitCode_list[i]+':'+str(i))

### divide time-related inputs, static inputs, and outputs
print('To tensor...')
total_instances = len(inputs)

# tr_input = torch.tensor(inputs[:][:3][:-1])
# st_input = torch.tensor(inputs[:][3:])
# output = torch.tensor(inputs[:][0][-1])


tr_input = []
st_input = []
output = []
for i in range(total_instances):
    tr_input.append(inputs[i][:3])
    st_input.append(inputs[i][3:-1])
    output.append(inputs[i][-1])
    # if i < 1:
        
    #     tr_input = torch.tensor(inputs[i][:3][:-1]).unsqueeze(0)
    #     # print(tr_input.size())
    #     st_input = torch.tensor(inputs[i][3:]).unsqueeze(0)
    #     output = torch.tensor(inputs[i][0][-1], dtype=torch.long).unsqueeze(0)
    # else:
    #     # print(torch.tensor(inputs[i][:3]).unsqueeze(0).size())
    #     tr_input = torch.cat((tr_input, torch.tensor(inputs[i][:3][:-1]).unsqueeze(0)))
    #     st_input = torch.cat((st_input, torch.tensor(inputs[i][3:]).unsqueeze(0)))
    #     output = torch.cat((output, torch.tensor(inputs[i][0][-1], dtype=torch.long).unsqueeze(0)))
    
    if i%50000 == 49999:
        print(str(round((i+1)*100/total_instances,2))+'%...')

tr_input = torch.tensor(tr_input, dtype=torch.double)
st_input = torch.tensor(st_input)#, dtype=torch.double)
output = torch.tensor(output, dtype=torch.long)

# print(tr_input[0,:,:])
# print(tr_input.size())
# print(st_input[0,:])
# print(st_input.size())
# print(output[0])
# print(output.size())

print('100%...')
### standardization
tr_input_sizes=tr_input.size()
# print(np.unique(tr_input[:,:,0].numpy()))
tr_input_flat = tr_input.view(-1, tr_input_sizes[-1])
# print(np.unique(tr_input_flat[:,0].numpy()))
tr_input_mean = tr_input_flat.mean(dim=0, keepdim=True)
tr_input_mean = tr_input_mean.view(1,1,tr_input_sizes[-1])
tr_input_std = tr_input_flat.std(dim=0, keepdim=True)
tr_input_std = tr_input_std.view(1,1,tr_input_sizes[-1])+0.0001
tr_input = (tr_input-tr_input_mean)/tr_input_std
tr_input = tr_input.float() #required by linear layer

st_input_mean = st_input.mean(dim=0, keepdim=True)
st_input_std = st_input.std(dim=0, keepdim=True)+0.0001
st_input = (st_input-st_input_mean)/st_input_std

# print(tr_input_mean)
# print(tr_input_std)
# print(st_input_mean)
# print(st_input_std)
# print(np.unique(output.numpy()))

### format for training and test
num_train = int(0.8*total_instances)
tr_input_train, tr_input_test = tr_input.split([num_train, total_instances-num_train])
st_input_train, st_input_test = st_input.split([num_train, total_instances-num_train])
output_train, output_test = output.split([num_train, total_instances-num_train])

train = tdata.TensorDataset(tr_input_train, st_input_train, output_train)
train_loader = tdata.DataLoader(train, batch_size=train_batch, shuffle=True)

test = tdata.TensorDataset(tr_input_test, st_input_test, output_test)
test_loader = tdata.DataLoader(test, batch_size=test_batch)

# print(tr_input_train.size())
# print(tr_input_test.size())
# print(st_input_train.size())
# print(st_input_test.size())
# print(output_train.size())
print(np.unique(output_test.numpy()))

print('Data ready...')

##############################################################################
#############    model construction
##############################################################################

class RNN(nn.Module):
    def __init__(self, tr_input_size, st_input_size, output_size, 
                 batch_size=64, hidden_size=3):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=tr_input_size, hidden_size=hidden_size, 
                          nonlinearity='relu', batch_first=True)

        self.fc = nn.Linear(hidden_size+st_input_size, output_size)


    def forward(self, tr_input, st_input):#, hidden):
        out_rnn, _ = self.rnn(tr_input, None)#hidden)
        last_tensor = torch.mean(out_rnn, dim=1)
        out = self.fc(torch.cat((st_input, last_tensor),dim=1))
        
        return out

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size)



### initialize a model
tr_input_size = tr_input_sizes[-1]
st_input_size = st_input.size()[-1]
output_size = len(ExitCode_list)
# print(output_size)

ex_rnn = RNN(tr_input_size, st_input_size, output_size, 
             batch_size = train_batch, hidden_size=2)

print('Model ready...')

##############################################################################
#############    training process
##############################################################################
clip=500
lr = 0.0001

### GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device using: ")
print(device)
ex_rnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ex_rnn.parameters(), lr=lr, momentum=0.9)

hidden = ex_rnn.initHidden()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        tr_input_i, st_input_i, output_i = data
        ### GPU
        tr_input_i, st_input_i, output_i = tr_input_i.to(device), st_input_i.to(device), output_i.to(device)
        output_i_r = ex_rnn(tr_input_i, st_input_i)#, hidden)
        loss = criterion(output_i_r, output_i)

        # ex_rnn.zero_grad()
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(ex_rnn.parameters(), clip)
        optimizer.step()

        # torch.nn.utils.clip_grad_norm_(ex_rnn.parameters(), clip)
        # for p in ex_rnn.parameters():
        #     if not torch.isnan(torch.max(p.data)):
        #         print(torch.max(p.data))
        #     if not torch.isnan(torch.max(p.grad.data)):
        #         print(torch.max(p.grad.data))
        #     p.data.add_(-lr, p.grad.data)
        # optimizer.step

        running_loss += loss.item()

        if i%1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


##############################################################################
#############    test process
##############################################################################

print('Testing starts...')

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         tr_input_i, st_input_i, out_put_i = data
#         ### GPU
#         tr_input_i, st_input_i, output_i = tr_input_i.to(device), st_input_i.to(device), output_i.to(device)
#         output_i_t = ex_rnn(tr_input_i, st_input_i)
#         _, predicted = torch.max(output_i_t, 1)
#         total += output_i.size(0)
#         correct += (predicted == output_i).sum().item()

# print('Accuracy of the network on the %d test images: %d %%' % (total, 
#     100 * correct / total))

class_correct = list(0. for i in range(output_size))
class_total = list(0. for i in range(output_size))
with torch.no_grad():
    for data in test_loader:
        tr_input_i, st_input_i, out_put_i = data
        ### GPU
        tr_input_i, st_input_i, output_i = tr_input_i.to(device), st_input_i.to(device), output_i.to(device)
        output_i_t = ex_rnn(tr_input_i, st_input_i)
        _, predicted = torch.max(output_i_t, 1)
        c = (predicted == output_i).squeeze()
        print(output_i)
        for i in range(test_batch):
            label = output_i[i]#.item()
            print(label)
            class_correct[label] += c[i].item()
            class_total[label] += 1
        # class_correct[output_i[0]] += c[0].item()
        # class_total[output_i[0]] += 1


for i in range(10):
    if class_total[i] > 0:
        print('Accuracy of %5s : %2d %%' % (
            ExitCode_list[i], 100 * class_correct[i] / class_total[i]))
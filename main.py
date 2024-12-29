import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math  
from stl_requirement_gen import gen_train_stl
from TraceGen import calculateDNF, simplifyDNF, trace_gen, monitor
from modeldef import RNNPredictor

import argparse

parser = argparse.ArgumentParser(description='Run experiment on data')
parser.add_argument('--data', type=str, default='step')
parser.add_argument('--timeunites', type=int, default=19)
parser.add_argument('--pastunites', type=int, default=5)
parser.add_argument('--seed', type=int, default=32)
parser.add_argument('--seed2', type=int, default=1)
parser.add_argument('--lambdao', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cell_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--noisep', type=float, default=0)
parser.add_argument('--percentage', type=float, default=0.7)
args = parser.parse_args()


BATCH_SIZE = 128
EPOCH = args.epochs  #85
TIMEUNITES = args.timeunites #20 
N_MC = 1 # iteration times 
LR = args.lr
lambda_o = args.lambdao
DROPOUT_RATE = 1
DROPOUT_TYPE = 4
START = 0
threshold = 100
nvar = 1
torch.manual_seed(args.seed)
aux = False
auxvar = 0

# past data points
PASTUNITES = args.pastunites # less than TIMEUNITES 
FUTUREUNITES = TIMEUNITES - PASTUNITES 

if args.data=='traffic1':
    trdataset = torch.tensor(np.loadtxt("traffic_trace_1.txt"))
    
    trdata = torch.stack((trdataset[0 :-1],trdataset[1 :]), dim=1)
    data_byunit = torch.zeros(trdata.size(0) - TIMEUNITES +1,TIMEUNITES,2)    
    for i in range(data_byunit.size(0)):
        data_byunit[i] = trdata[i:i+TIMEUNITES]
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]

elif args.data=='traffic2':
    trafficdata = torch.tensor(np.loadtxt("traffic_trace_2.txt"))
    
    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, 0] = trafficdata[j, START:START+TIMEUNITES]
        data_byunit[j, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1]
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
    
elif args.data=='airmulti':
    aux = True
    beijing_pm25 = torch.load('beijing_pm25.dat')
    #print(beijing_pm25.shape)  #torch.Size([36, 8760])
    beijing_pm25_dt = beijing_pm25.view(beijing_pm25.size(0), -1, 24)   #beijing_pm25_dt: torch.Size([36, 365, 24])
    
    beijing_pm25_dt[1:] = beijing_pm25_dt[1:] - beijing_pm25_dt[0]
    nvar = beijing_pm25_dt.size(0)
    #print(nvar)  # 36
    data_byunit = beijing_pm25_dt.transpose(0, 1).transpose(1, 2)
    #print(data_byunit.shape)    #torch.Size([365, 24, 36])
    data_byunit = torch.stack((data_byunit[:, :-1, :].clone(), data_byunit[:, 1:, :].clone()), dim=-1)      # pred next step
    #print(data_byunit.shape)    #torch.Size([365, 23, 36, 2])
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]        
    #print(data_byunit.shape)   #torch.Size([365, 23, 36, 2]) 
    
    if TIMEUNITES<23:
        data_byunit = data_byunit[:, :TIMEUNITES, :, :]
        
    if aux:
        auxdata = torch.eye(TIMEUNITES).unsqueeze(0).expand(data_byunit.size(0), TIMEUNITES, TIMEUNITES).float() # identity matrix
        auxvar = TIMEUNITES
        #print(auxdata.shape) # torch.Size([365, 23, 23])
        #print(auxvar) # 23

elif args.data=='airpde':
    aux = True
    beijing_pm25_train = np.load('beijing_data_new/train.npz')
    beijing_pm25_val = np.load('beijing_data_new/val.npz')
    beijing_pm25_test = np.load('beijing_data_new/test.npz')
    
    x_train = beijing_pm25_train['x'] #(2180, 24, 35, 6)    seq_len:: 24
    y_train = beijing_pm25_train['y'] #(2180, 24, 35, 6)    
    x_val = beijing_pm25_val['x'] #(311, 24, 35, 6)
    y_val = beijing_pm25_val['y'] #(311, 24, 35, 6)
    x_test = beijing_pm25_test['x'] #(623, 24, 35, 6)  
    y_test = beijing_pm25_test['y'] #(623, 24, 35, 6)
    
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    x = np.concatenate((x, x_test), axis=0) #(3114, 24, 35, 6) 
    y = np.concatenate((y, y_test), axis=0) #(3114, 24, 35, 6) 
    
    x = torch.tensor(x)
    y = torch.tensor(y)
    x = torch.reshape(x, (3114, 24, -1))
    y = torch.reshape(y, (3114, 24, -1))
    
    beijing_pm25_dt = x
    
    nvar = int(beijing_pm25_dt.shape[-1]/6)  # 210 = 35*6  number of feature, 35 nodes * 6 features
    data_byunit = torch.stack((x.clone().float(), y.clone().float()), dim=-1)
    #data_byunit = torch.stack((x[:, :-1, :].clone(), y[:, 1:, :].clone()), dim=-1)
    #print(data_byunit.size())   #torch.Size([3114, 24, 210, 2])    210 = 35*6
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
    #print(data_byunit.shape)   #torch.Size([3114, 24, 210, 2])   
    
    if TIMEUNITES<23:
        data_byunit = data_byunit[:, :TIMEUNITES, :, :]
        #print(data_byunit.shape)
    if aux:
        auxdata = torch.eye(TIMEUNITES).unsqueeze(0).expand(data_byunit.size(0), TIMEUNITES, TIMEUNITES).float() # identity matrix       
        #print(auxdata.size()) # torch.Size([3114, 24, 24])
        auxvar = TIMEUNITES
        #print(auxvar) # 24

        
elif args.data=='multi' or args.data=='multijump' or args.data=='multistep' or args.data=='multieven' or args.data=='unusual' or args.data=='consecutive':
    a = np.loadtxt("generate_iterative_%s.dat" % args.data)
    # amount of data used, part
    trafficdata = torch.tensor(a)
    nvar=2
    trafficdata = trafficdata.view(trafficdata.size(0), -1, nvar)
    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, nvar, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, :, 0] = trafficdata[j, START:START+TIMEUNITES, :]
        data_byunit[j, :, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1, :]
    if args.data!='unusual':
        data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]
    
elif args.data!='air':
    a = np.loadtxt("generate_iterative_%s.dat" % args.data)
    # amount of data used, part
    trafficdata = torch.tensor(a)

    data_byunit = torch.zeros(trafficdata.size(0), TIMEUNITES, 2)
    for j in range(trafficdata.size(0)):
        data_byunit[j, :, 0] = trafficdata[j, START:START+TIMEUNITES]
        data_byunit[j, :, 1] = trafficdata[j, START+1:START+TIMEUNITES+1]
    

    data_byunit = data_byunit.unsqueeze(2)
    # Shuffle data
    data_byunit = data_byunit[torch.randperm(data_byunit.size(0))]  #why random perm


torch.manual_seed(args.seed2)
train_split = args.percentage
#print(train_split) #0.95

td = data_byunit[:int(train_split * data_byunit.size(0))]   
#print(td.size())   

if args.noisep > 0:
    noise_bool = (torch.rand(td.size()) > args.noisep).float()
    td = td * noise_bool
if aux:
    train_data = torch.utils.data.TensorDataset(td, auxdata[:int(train_split * data_byunit.size(0))])   
    #torch.Size([346, 23, 36, 2])       torch.Size([346, 23, 23])
    test_data = torch.utils.data.TensorDataset(data_byunit[int(train_split * data_byunit.size(0)):], auxdata[int(train_split * data_byunit.size(0)):])    
    #torch.Size([19, 23, 36, 2])       torch.Size([19, 23, 23])
else:
    train_data = torch.utils.data.TensorDataset(td)
    test_data = torch.utils.data.TensorDataset(data_byunit[int(train_split * data_byunit.size(0)):])
       


train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

#print(TIMEUNITES)      #24
#print(PASTUNITES)      #5

#print(gen_train_stl(args.data, TIMEUNITES, PASTUNITES)) #string
if args.data =='airpde':
    DNFs = calculateDNF(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, True, TIMEUNITES, 35)
else:   
    DNFs = calculateDNF(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, True, TIMEUNITES, nvar)
    
DNFs = simplifyDNF(DNFs)
DNFs = torch.FloatTensor(DNFs)



# bigger batch size, smaller learning rate 

# training
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, datasample in tqdm(enumerate(train_loader)):
        if aux:
            if args.data=='airpde':                
                data, auxdata = datasample[0][:, :, :, 0], datasample[1]
                target = datasample[0][:, :, :, 1]  #[3114, 24, 234]
                target = target[:,:,:210]       #torch.Size([128, 24, 210])   
                target = torch.reshape(target, (-1, 24, 35, 6))
                target = target[:, :, :, 0] #[128, 24, 35]
                
            else:
                data, auxdata, target = datasample[0][:, :, :, 0], datasample[1], datasample[0][:, :, :, 1] 
                #print(datasample[0].shape)  #torch.Size([128, 23, 36, 2])
                
            data, auxdata, target = data.to(device), auxdata.to(device), target.to(device)
        else:
            if args.data=='airpde':
                data = datasample[0][:, :, :, 0]
                target = datasample[0][:, :, :, 1]  #[128, 24, 210]
                target = torch.reshape(target, (128, 24, 35, 6))
                target = target[:, :, :, 0] #[128, 24, 35]
            else:
                data, target = datasample[0][:, :, :, 0], datasample[0][:, :, :, 1]
                
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if aux:
            #print(data.size())     #torch.Size([128, 24, 35, 6])
            #print(auxdata.size())      #torch.Size([128, 24, 24])
            output = model((data, auxdata))     
        else:
            output = model(data)
        t = 0
        if lambda_o!=0:
            #print(DNFs.size())  #torch.Size([1, 24, 210, 2])
            loss = criterion(output, target) + lambda_o * criterion(output, trace_gen(DNFs, output.cpu()).to(device))  
        else:
            loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    print("train loss: %.3f" %train_loss)    # MSE with ground truth + regularization
    #print("len: %.3f" %len(train_loader))
    print(" ")



# testing 
def test(model, device, test_loader, criterion, epoch):
    test_loss = 0
    t_loss_2 = 0
    test_loss_mean = 0
    test_loss_t = 0
    t_loss_2_t = 0
    test_loss_mean_t = 0
    satisfy =  .0
    satisfy_t = .0
    traceset = []
    with torch.no_grad():
        for batch_idx, datasample in tqdm(enumerate(test_loader)):
            
            if datasample[0].size(0) != BATCH_SIZE: 
                current_batch_size = datasample[0].size(0)
            else:
                current_batch_size = BATCH_SIZE
            
            if aux:
                if args.data=='airpde':                
                    data, auxdata = datasample[0][:, :, :, 0], datasample[1]
                    target = datasample[0][:, :, :, 1]  #[3114, 24, 234]
                    target = target[:,:,:210]       #torch.Size([128, 24, 210])   
                    target = torch.reshape(target, (-1, 24, 35, 6))
                    target = target[:, :, :, 0] #[128, 24, 35]
                    
                else:
                    data, auxdata, target = datasample[0][:, :, :, 0], datasample[1], datasample[0][:, :, :, 1] 
                    
                data, auxdata, target = data.to(device), auxdata.to(device), target.to(device)
            else:
                if args.data=='airpde':
                    data = datasample[0][:, :, :, 0]
                    target = datasample[0][:, :, :, 1]  #[128, 24, 210]
                    target = torch.reshape(target, (128, 24, 35, 6))
                    target = target[:, :, :, 0] #[128, 24, 35]
                else:
                    data, target = datasample[0][:, :, :, 0], datasample[0][:, :, :, 1]
                    
                data, target = data.to(device), target.to(device)
            
            
#            if aux:
#                data, auxdata, target = datasample[0][:, :, :, 0], datasample[1], datasample[0][:, :, :, 1]
#                data, auxdata, target = data.to(device), auxdata.to(device), target.to(device)
#            else:
#                data, target = datasample[0][:, :, :, 0], datasample[0][:, :, :, 1]
#                data, target = data.to(device), target.to(device)
            output = torch.zeros(N_MC, current_batch_size, FUTUREUNITES, nvar)
            teacher_trace = torch.zeros(N_MC, current_batch_size, FUTUREUNITES, nvar)
            
            
            
            for i in range(N_MC):   # N_MC = 1
                if aux:
                    output[i] = model.forward_test_with_past((data, auxdata))
                else:
                    output[i] = model.forward_test_with_past(data)                   
                #print(output[i].size())     #torch.Size([128, 19, 210])
                
                test_loss += criterion(output[i], target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()        # MSE with gronnd truth
                robustness = monitor(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, torch.cat((target[:, :PASTUNITES, :].cpu(), output[i]), dim=1))
                t_loss_2 += F.relu(-robustness).sum().item()    #get all negative
                satisfy += (robustness >= 0).sum().item()       #get all positive
                
                teacher_trace[i] = trace_gen(DNFs, torch.cat((target[:, :PASTUNITES, :].cpu(), output[i]), dim=1))[:, PASTUNITES:, :]                
                test_loss_t += criterion(teacher_trace[i], target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()   # MSE teacher_trace with gronnd truth
                robustness_t = monitor(gen_train_stl(args.data, TIMEUNITES, PASTUNITES), 0, torch.cat((target[:, :PASTUNITES, :].cpu(), teacher_trace[i]), dim=1))
                t_loss_2_t += F.relu(-robustness_t).sum().item()
                satisfy_t += (robustness_t >= 0).sum().item()

            if epoch == EPOCH:
                trace = torch.stack((output.mean(dim = 0), output.std(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu(), torch.zeros(output.mean(dim = 0).size())), dim = -1)
                traceset.append(trace)
            test_loss_mean += criterion(output.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
            test_loss_mean_t += criterion(teacher_trace.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES, :].cpu()).item()
            
        if epoch == EPOCH:    
            traceset = torch.cat(traceset, dim = 0)  

        satisfy /= len(test_loader.dataset) * N_MC
        satisfy_t /= len(test_loader.dataset) * N_MC
        n = len(test_loader) 
        
        #print("len: %.3f" %len(test_loader))       
        print("student test loss: %.3f" %(test_loss/n))
        #print("test loss_mean: %.3f" %test_loss_mean)
        print("student robustness loss: %.3f" %(t_loss_2/n))       
        print("student satisfy: %.2f%%" % (satisfy * 100))
        print(" ")
        print("teacher test loss: %.3f" % (test_loss_t/n))
        #print("teacher test loss_mean: %.3f" % test_loss_mean_t)
        print("teacher robustness loss: %.3f" %(t_loss_2_t/n))
        print("teacher satisfy %.2f%%" % (satisfy_t * 100))
        
    if epoch==EPOCH:
        f1 = open('record_res_%s.txt' % args.data,'a')
        f1.write('%s %.2f %d %d %.2f %.4f %.4f %.4f %.4f %.2f%% %.2f%%\n' % (args.data, args.lr, PASTUNITES, args.seed, lambda_o, test_loss, t_loss_2, test_loss_t, t_loss_2_t, satisfy * 100, satisfy_t * 100))
        f1.close()

model = RNNPredictor(nvar=nvar, auxvar=auxvar, cell_type=args.cell_type, dropout_type=DROPOUT_TYPE, future_unit=FUTUREUNITES, past_unit=PASTUNITES)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = LR )

# run 
for epoch in range(1, EPOCH+1):
    print("==============================================" )
    print("Epoch: %.0f" %epoch)
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion, epoch)
    print(" " )

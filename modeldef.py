import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn._VF as _VF
import math
import numpy as np
import torch.nn.init as init
# input [batch_size per batch (128, 1024), time_units, input_dimension]
class RNNPredictor(nn.Module):
    # initial function
    def __init__(self, cell_type='lstm', nvar=1, auxvar=0, hidden_size=512, dropout_rate=1, dropout_type=1, future_unit=19, past_unit=5):
        super(RNNPredictor, self). __init__()
        self.hidden_size = hidden_size
        if cell_type == 'lstm' or 'pde':
            self.lstm = nn.LSTMCell(auxvar + nvar, self.hidden_size)
        elif cell_type == 'gru':
            self.lstm = nn.GRUCell(auxvar + nvar, self.hidden_size)
        else:
            self.lstm = nn.RNNCell(auxvar + nvar, self.hidden_size)
            
            
        self.cell_type = cell_type
        self.linear = nn.Linear(self.hidden_size, nvar)
        self.linear_pde = nn.Linear(self.hidden_size, 35)    #[in feature, out feature] #airpde dataset sensors
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.future_unit = future_unit
        self.past_unit = past_unit
        self.nvar = nvar
        self.auxvar = auxvar

        #------- pde parameter -------
        self.scale1 = nn.Parameter(torch.FloatTensor(1))
        init.normal_(self.scale1, mean=0.5, std=0.1)
        self.scale2 = nn.Parameter(torch.FloatTensor(1))  
        init.normal_(self.scale2, mean=0.5, std=0.1)
        self.scale3 = nn.Parameter(torch.FloatTensor(1))  
        init.normal_(self.scale3, mean=0.5, std=0.1)
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.beta = nn.Parameter(torch.FloatTensor(1))       
        init.normal_(self.alpha, mean=0.5, std=0.1)    
        init.normal_(self.beta, mean=0.5, std=0.1)      
        self.h = nn.Parameter(torch.FloatTensor(35))    #airpde dataset sensors
        init.normal_(self.h, mean=5, std=0.1) 

        
    #pde layer
    def pde_layer(self, input, speed, direction, adj_matrix):    
                          

        #h = nn.Parameter(torch.FloatTensor(input.size(2))).to(input.device)
        #init.normal_(h, mean=5, std=0.5)   
        
        
        h = self.h
        #h = self.h[:input.size(0), :input.size(1), :]  #get subarray of to meet the size of input
       
        weight_matrix = adj_matrix        
        weight_matrix[weight_matrix < 0.001] = 0
        weight_matrix = torch.from_numpy(weight_matrix).type(torch.FloatTensor).to(input.device)
        
        pm25 = input    #torch.Size([128, 24, 35])
        v_x = speed*torch.cos(direction)    #torch.Size([128, 24, 35])
        v_y = speed*torch.sin(direction)    #torch.Size([128, 24, 35])
        source_matrix = torch.transpose(weight_matrix, 0, 1)
        
        #print(pm25.size())        
        
        D = torch.sum(weight_matrix, dim = 1)
        D = torch.diag(D)
        L = D - weight_matrix
        k = 0.3
        
        #print(pm25)
        delta_t  = 3 
        
        for t in range(0, 6): #pm25.size(1)):  #time step, hyper perameter
            
            #-----------  divergence  -----------
            diff_flow_x = torch.matmul(v_x *pm25, (source_matrix - weight_matrix)) 
            diff_flow_y = torch.matmul(v_y *pm25, (source_matrix - weight_matrix))        
            div_f = diff_flow_x + diff_flow_y
            
            #-----------  diffusion  -----------   
            laplacian = k*torch.matmul(pm25, -L)
            
            pde_results = pm25 + self.scale1*(-div_f)*delta_t + self.scale2*laplacian*delta_t + self.scale3*h*delta_t
            pm25 = pde_results
            
        pde_results = torch.relu(pde_results)
        
        #-----------  resnet  -----------          
        outputs = self.alpha * input + self.beta * pde_results # only update pm25

        return outputs
    
    # forward function
    def forward(self, input):
        
        first = input[0]
        if first.size(-1) >= 210:   #airpde dataset
            curr_input = input[0][:, :, :210]
            curr_input = torch.reshape(curr_input, (-1, 24, 35, 6))
            first = curr_input[:, :, :, 0]
            speed = curr_input[:, :, :, -1]
            direction = curr_input[:, :, :, -2]
            
        if self.auxvar>0:
            input = torch.cat((first, input[1]), dim=-1)           
        else:
            input = first    
        outputs = []
        
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        if self.cell_type == 'lstm' or 'pde':
            c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
       
                                
        for i in range(input.size(1)):  # i:time step
            
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input[:, i, :], (h_t, c_t))
            else:
                h_t = self.lstm.forward(input[:, i, :], h_t)
            
            feature = input[:, i, :].size(-1)
            if feature == 35:     #airpde dataset
                output = self.linear_pde(h_t)   #h_t: torch.Size([128, 512]),   output: torch.Size([128, 35])             
            else:
                output = self.linear(h_t)       #h_t: torch.Size([128, 512]),   output: torch.Size([128, 36])
               
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)   #torch.Size([128, 24, 210])    #make this small to 35 not 210, or reduce dimension
        
        
        if self.cell_type == 'pde':
            adj_matrix = np.load('adj_matrix_beijing.npy')           
            outputs = self.pde_layer(outputs, speed, direction, adj_matrix)
            
        return outputs

    def forward_test(self, input):
        
        first = input[0]
        if first.size(-1) >= 210:   #airpde dataset
            curr_input = input[0][:, :, :210]
            curr_input = torch.reshape(curr_input, (-1, 24, 35, 6))
            first = curr_input[:, :, :, 0]
            speed = curr_input[:, :, :, -1]
            direction = curr_input[:, :, :, -2]
            
        if self.auxvar>0:
            auxinput = input[1]
            
        input = first

        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        if self.cell_type == 'lstm' or 'pde':
            c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :]
        
        for i in range(input.size(1)):
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i, :]), dim=1)
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
                      
            feature = input[:, i, :].size(-1)
            if feature == 35:     #airpde dataset
                output = self.linear_pde(h_t)   #h_t: torch.Size([128, 512]),   output: torch.Size([128, 35])             
            else:
                output = self.linear(h_t)       #h_t: torch.Size([128, 512]),   output: torch.Size([128, 210])
                                
            input_local = output
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        
        if self.cell_type == 'pde':
            adj_matrix = np.load('adj_matrix_beiijng.npy')           
            outputs = self.pde_layer(outputs, speed, direction, adj_matrix)
            
        return outputs
        
    
    def forward_test_with_past(self, input):
        
        first = input[0]
        if first.size(-1) >= 210:   #airpde dataset
            curr_input = input[0][:, :, :210]
            curr_input = torch.reshape(curr_input, (-1, 24, 35, 6))
            first = curr_input[:, :, :, 0]
            speed = curr_input[:, self.past_unit:, :, -1]
            direction = curr_input[:, self.past_unit:, :, -2]
            
            
        if self.auxvar>0:
            auxinput = input[1]
            
        input = first
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        if self.cell_type == 'lstm' or 'pde':
            c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :] 
        
        for i in range(self.past_unit+1):   # pred with input
            if self.auxvar>0:
                input_t = torch.cat((input[:, i, :], auxinput[:, i, :]), dim=1)
            else:
                input_t = input[:, i, :]
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_t, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_t, h_t)
            
            feature = input[:, i, :].size(-1)
            if feature == 35:     
                output = self.linear_pde(h_t)   #h_t: torch.Size([128, 512]),   output: torch.Size([128, 35])             
            else:
                output = self.linear(h_t)       #h_t: torch.Size([128, 512]),   output: torch.Size([128, 210])
                                
            input_local = output
        
        outputs.append(input_local)   
        
        for i in range(self.future_unit-1):     # pred with pseudo input
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i + self.past_unit + 1, :]), dim=1)
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
                
            feature = input[:, i, :].size(-1)
            if feature == 35:        
                output = self.linear_pde(h_t)   #h_t: torch.Size([128, 512]),   output: torch.Size([128, 35])             
            else:
                output = self.linear(h_t)       #h_t: torch.Size([128, 512]),   output: torch.Size([128, 210])
                                
            input_local = output
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)   #torch.Size([128, 19, 35])   
        #print(outputs.size()) #torch.Size([128, 19, 35])
        if self.cell_type == 'pde':
            adj_matrix = np.load('adj_matrix_beijing.npy')           
            outputs = self.pde_layer(outputs, speed, direction, adj_matrix)
            
        return outputs
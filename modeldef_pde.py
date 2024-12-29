import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn._VF as _VF
import math
    
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
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.future_unit = future_unit
        self.past_unit = past_unit
        self.nvar = nvar
        self.auxvar = auxvar

    #pde layer
    def pde_layer(self, input, adj_matrix, delta_t):
        
        #for i in range(input.size(1)):        
        #    diff_flow_x = v_x *(down-up)*input[:, i, :]
        #    diff_flow_y = v_y *(down-up)*input[:, i, :]
        #    diff_flow_z = v_z *(down-up)*input[:, i, :]        
        #    div_f = diff_flow_x + diff_flow_y + diff_flow_z
            
        #    k = torch.variable()
        #    laplacian = k *(down- 2 + up)*input[:, i, :]

        #    pde_result = -div_f*delta_t + laplacian*delta_t + input[:, i, :]    
        #    pde_results.append(pde_result)
            
        
        diff_flow_x = v_x *(down-up)*input
        diff_flow_y = v_y *(down-up)*input
        diff_flow_z = v_z *(down-up)*input        
        div_f = diff_flow_x + diff_flow_y + diff_flow_z
            
        k = nn.Parameter(torch.FloatTensor(1).to(self.device))
        laplacian = k *(down- 2 + up)*input

        pde_results = -div_f*delta_t + laplacian*delta_t + input    
        
        
        alpha = nn.Parameter(torch.FloatTensor(1).to(self.device))
        beta = nn.Parameter(torch.FloatTensor(1).to(self.device))
        
        outputs = alpha * input + beta * pde_results 
        
        return outputs
    
    # forward function
    def forward(self, input):
        if self.auxvar>0:
            input = torch.cat((input[0], input[1]), dim=-1)
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        if self.cell_type == 'lstm' or 'pde':
            c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        

        for i in range(input.size(1)):  # [time step]
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input[:, i, :], (h_t, c_t))
            else:
                h_t = self.lstm.forward(input[:, i, :], h_t)
            output = self.linear(h_t)
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        print(outputs.size())
        if self.cell_type == 'pde':
            outputs = pde_layer(outputs, adj_matrix)
            
        return outputs

    def forward_test(self, input):
        if self.auxvar>0:
            auxinput = input[1]
            input = input[0]
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :]
        
        for i in range(input.size(1)):
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i, :]), dim=1)
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
            
            output = self.linear(h_t)
            input_local = output
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        return outputs
        
    
    def forward_test_with_past(self, input):
        if self.auxvar>0:
            auxinput = input[1]
            input = input[0]
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float, device=input.device)
        
        input_local = input[:, 0, :] 
        
        for i in range(self.past_unit+1):
            if self.auxvar>0:
                input_t = torch.cat((input[:, i, :], auxinput[:, i, :]), dim=1)
            else:
                input_t = input[:, i, :]
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_t, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_t, h_t)
            output = self.linear(h_t)
            input_local = output
        
        outputs.append(input_local)   
        for i in range(self.future_unit-1):
            if self.auxvar>0:
                input_local = torch.cat((input_local, auxinput[:, i + self.past_unit + 1, :]), dim=1)
            if self.cell_type == 'lstm' or 'pde':
                h_t, c_t = self.lstm.forward(input_local, (h_t, c_t))
            else:
                h_t = self.lstm.forward(input_local, h_t)
            output = self.linear(h_t)
            input_local = output
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs
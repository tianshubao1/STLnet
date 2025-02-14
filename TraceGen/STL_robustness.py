import numpy as np
import torch

adj = np.load('adj_matrix_beijing.npy')


def monitor(req, t, signal):        #this is for testing
    if req[0] == "mu":          #req: ["mu", location, value]
        rho = signal[:, t, req[1]] - req[2]     #pred: [batch, time, location], pred - trace_gen. If satisfy, this is positive
        #print(rho)
        return rho
    elif req[0] == "neg":
        rho = -monitor(req[1], t, signal)
        return rho
    elif req[0] == "and":
        rho = torch.min(monitor(req[1], t, signal), monitor(req[2], t, signal))
        return rho
    elif req[0] == "or":
        rho = torch.max(monitor(req[1], t, signal), monitor(req[2], t, signal))
        return rho
    elif req[0] == "always":
        t1 = req[1][0]
        t2 = req[1][1]
        rho = monitor(req[2], t+t1,  signal)
        for ti in range(t1, t2+1):
            rho = torch.min(rho, monitor(req[2], t+ti,  signal))
        return rho
    elif req[0] == "eventually":
        t1 = req[1][0]
        t2 = req[1][1]
        rho = monitor(req[2], t+t1,  signal)
        for ti in range(t1, t2+1):
            rho = torch.max(rho, monitor(req[2], t+ti,  signal))
        return rho
    elif req[0] == "until":
        t1 = req[1][0]
        t2 = req[1][1]
        rho = monitor(req[3], t+t1,  signal)
        for ti in range(t1, t2+1):      #[t1, ti, t2]
            rho1 = monitor(req[3], t+ti, signal)
            rho2 = monitor(req[2], t+ti, signal)
            for tj in range(t1, ti):
                rho2 = torch.min(rho2, monitor(req[2], t+tj,  signal))
            rho3 = torch.min(rho2, rho1)
            rho = torch.max(rho, rho3)
        return rho
        
    elif req[0] == "surround":  #phi_spot
        pivot = req[1]    #curr location
        d1 = req[2][0]      #distance min
        d2 = req[2][1]      #distance max
        h_in = req[3]       # <= h_in
        h_out = req[4]      # >= h_out
        
        #rho = monitor( ('mu', pivot, h_in), t, signal)   #calculate pivot          >= h_in 
        rho = monitor( ('neg',('mu', pivot, h_in)), t, signal)   #calculate pivot          <= h_in 
        for node in range(0, signal.size(2)):   #signal:[batch, time, location]
            if adj[node, pivot] < d1 and node!= pivot:   #specify node in phi
                #rho_in_curr = monitor( ('mu', node, h_in), t, signal)    #h_in is the lower bound,   >=h_in
                rho_in_curr = monitor( ('neg',('mu', node, h_in)), t, signal)    #h_in is the upper bound,   <=h_in
                rho = torch.min(rho, rho_in_curr)   
                    
            elif adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                #rho_out_curr = monitor(('neg',('mu', node, h_out)), t, signal)    #h_out is the upper bound, <= h_out
                rho_out_curr = monitor((('mu', node, h_out)), t, signal)    #h_out is the lower bound, >= h_out
                rho = torch.min(rho, rho_out_curr)            
        return rho    
        
    elif req[0] == "somewhere":
        
        pivot = req[1]    #curr location
        d1 = req[2][0]      #distance min
        d2 = req[2][1]      #distance max
        h = req[3]       
        rho = monitor(('neg', ('mu', pivot, h)), t, signal)   #calculate pivot          <= h_in 
        
        for node in range(0, signal.size(2)):   #signal:[batch, time, location]
            if adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                rho_out_curr = monitor(('neg',('mu', node, h)), t, signal)    #nearby value cant be too high, <= h
                rho = torch.min(rho, rho_out_curr)

        return rho           

            
if __name__=='__main__':
    always_multi_req = ('always', (0, 1), ('and', ('and', ('mu', 0, -1.0), ('neg', ('mu', 0, 1.0))), ('and', ('mu', 1, -1.0), ('neg', ('mu', 1, 1.0)))))
    print(monitor(always_multi_req, 0, torch.FloatTensor([[[-1,0], [-1,0]], [[-1,-1], [-0.5,0.5]]]).float()))

import numpy as np
import torch

maxf = 20181028
minf = -20181028

adj = np.load('adj_matrix_beijing.npy')
#adj = np.load('adj_matrix.npy')        #for local test

def simplifyDNF(DNFs):
    t = DNFs.size(1)
    DNFs = DNFs.view(DNFs.size(0), -1, DNFs.size(-1))
    i = 0
    min_signal = (DNFs.unsqueeze(1)[:, :, :, 0] <= DNFs.unsqueeze(0)[:, :, :, 0]+1e-4).min(dim=2)[0]
    max_signal = (DNFs.unsqueeze(1)[:, :, :, 1] >= DNFs.unsqueeze(0)[:, :, :, 1]-1e-4).min(dim=2)[0]
    flags = torch.ones(max_signal.size(0)).bool()
    for i in range(max_signal.size(0)):
        if flags[i].item():
            for j in range(max_signal.size(1)):
                if j!=i and max_signal[i,j].item() and min_signal[i,j].item():
                    flags[j] = 0
    DNFs = DNFs[flags.nonzero().view(-1)]
    DNFs = DNFs.view(DNFs.size(0), t, -1, DNFs.size(-1))
    return DNFs
    
def calculateDNF(req, t, flag, T=100, NV=1):
    if not flag:
        if req[0] == "mu":
            s = torch.full((1, T, NV, 2), maxf)
            s[0, :, :, 0] = -s[0, :, :, 0]
            s[0, t, req[1], 1] = req[2]     #upper bound is req[2]
            return s
        elif req[0] == "neg":
            return calculateDNF(req[1], t, True, T, NV)
        elif req[0] == "and":
            return calculateDNF(('or', req[1], req[2]), t, True, T, NV)
        elif req[0] == "or":
            return calculateDNF(('and', req[1], req[2]), t, True, T, NV)
        elif req[0] == "always":
            return calculateDNF(('eventually', req[1]), t, True, T, NV)
        elif req[0] == "eventually":
            return calculateDNF(('always', req[1]), t, True, T, NV)
        elif req[0] == "everywhere":
            return calculateDNF(('somewhere', req[1]), t, True, T, NV)
            
        elif req[0] == "surround":  
            pivot = req[1]    #curr location
            d1 = req[2][0]      #distance min
            d2 = req[2][1]      #distance max
            h_in = req[3]
            h_out = req[4]
            
            s1 = calculateDNF(('mu', pivot, h_in), t, False, T, NV)   #calculate pivot <= h_in
            for node in range(0, NV):
                if adj[node, pivot] < d1 and node!= pivot:   #specify node in phi
                    set_in_curr = calculateDNF(('mu', node, h_in), t, False, T, NV)     #<= h_in is the upper bound
                    sc = torch.zeros((s1.shape[0], set_in_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s1[:, :, :, 0].unsqueeze(1), set_in_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s1[:, :, :, 1].unsqueeze(1), set_in_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s1 = sc
                    
                elif adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                    set_out_curr = calculateDNF(('mu', node, h_out), t, False, T, NV)    #<= h_out is the lower bound
                    sc = torch.zeros((s1.shape[0], set_out_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s1[:, :, :, 0].unsqueeze(1), set_out_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s1[:, :, :, 1].unsqueeze(1), set_out_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s1 = sc
            s1 = simplifyDNF(s1)

            s2 = calculateDNF(('mu', pivot, h_in), t, True, T, NV)   #calculate pivot >= h_in
            for node in range(0, NV):
                if adj[node, pivot] < d1 and node!= pivot:   #specify node in phi
                    set_in_curr = calculateDNF(('mu', node, h_in), t, True, T, NV)     #>= h_in is the upper bound
                    sc = torch.zeros((s2.shape[0], set_in_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s2[:, :, :, 0].unsqueeze(1), set_in_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s2[:, :, :, 1].unsqueeze(1), set_in_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s2 = sc
                    
                elif adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                    set_out_curr = calculateDNF(('mu', node, h_out), t, True, T, NV)    #>= h_out is the lower bound
                    sc = torch.zeros((s2.shape[0], set_out_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s2[:, :, :, 0].unsqueeze(1), set_out_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s2[:, :, :, 1].unsqueeze(1), set_out_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s2 = sc
            s2 = simplifyDNF(s2)
            
            s3 = calculateDNF(('mu', pivot, h_in), t, False, T, NV)   #calculate pivot <= h_in
            for node in range(0, NV):
                if adj[node, pivot] < d1 and node!= pivot:   #specify node in phi
                    set_in_curr = calculateDNF(('mu', node, h_in), t, False, T, NV)     #<= h_in is the upper bound
                    sc = torch.zeros((s3.shape[0], set_in_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s3[:, :, :, 0].unsqueeze(1), set_in_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s3[:, :, :, 1].unsqueeze(1), set_in_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s3 = sc
                    
                elif adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                    set_out_curr = calculateDNF(('mu', node, h_out), t, True, T, NV)    #>= h_out is the lower bound
                    sc = torch.zeros((s3.shape[0], set_out_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s3[:, :, :, 0].unsqueeze(1), set_out_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s3[:, :, :, 1].unsqueeze(1), set_out_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s3 = sc
            s3 = simplifyDNF(s3)
            
            s = torch.cat((s1, s2, s3), dim=0)
            
            return s #simplifyDNF(s)
            
            
        #add "somewhere"    
            
    else:
        if req[0] == "mu":
            s = torch.full((1, T, NV, 2), maxf)     #[T, NV, 2], upper bound is inf, lower bound is -inf
            s[0, :, :, 0] = -s[0, :, :, 0]  # min value
            s[0, t, req[1], 0] = req[2]     # at time t, varible location req[1], low bound is req[2] to inf 
            return s
        elif req[0] == "neg":
            return calculateDNF(req[1], t, False, T, NV)
        elif req[0] == "and":
            set1 = calculateDNF(req[1], t, True, T, NV)
            set2 = calculateDNF(req[2], t, True, T, NV)
            s = torch.zeros((set1.shape[0], set2.shape[0], T, NV, 2))       #[1, 1, T, NV, 2]
            s[:, :, :, :, 0] = torch.max(set1[:, :, :, 0].unsqueeze(1), set2[:, :, :, 0].unsqueeze(0))      #min, compare min of the 2, pick the max of them
            s[:, :, :, :, 1] = torch.min(set1[:, :, :, 1].unsqueeze(1), set2[:, :, :, 1].unsqueeze(0))      #max, compare max of the 2, pick the min of them
            s = s.view(-1, T, NV, 2)    #merge first 2 dimension
            s = simplifyDNF(s)
            return s
        elif req[0] == "or":
            set1 = calculateDNF(req[1], t, True, T, NV)
            set2 = calculateDNF(req[2], t, True, T, NV)
            return torch.cat((set1, set2), dim=0)
        elif req[0] == "always":
            t1 = req[1][0]      #start
            t2 = req[1][1]      #end
            s = calculateDNF(req[2], t + t1, True, T, NV)   #check req step by step
            for tt in range(t + t1 + 1, t + t2 + 1):
                set_curr = calculateDNF(req[2], tt, True, T, NV)
                sc = torch.zeros((s.shape[0], set_curr.shape[0], T, NV, 2))     #merge
                sc[:, :, :, :, 0] = torch.max(s[:, :, :, 0].unsqueeze(1), set_curr[:, :, :, 0].unsqueeze(0))
                sc[:, :, :, :, 1] = torch.min(s[:, :, :, 1].unsqueeze(1), set_curr[:, :, :, 1].unsqueeze(0))
                sc = sc.view(-1, T, NV, 2)
                sc = simplifyDNF(sc)
                s = sc
            return s
        elif req[0] == "eventually":
            t1 = req[1][0]
            t2 = req[1][1]
            s = torch.zeros((0, T, NV, 2))
            for tt in range(t + t1, t + t2 + 1):
                set_curr = calculateDNF(req[2], tt, True, T, NV)
                s = torch.cat((s, set_curr), dim=0)
            return s

        #to do        
        elif req[0] == "surround":      
            #need to specify a location, specift time first so dont need to consider time
            #("surround", 5, (0.2, 0.6), h_in, h_out), need to specify weak spot and exclude strong spots, phi_1: x <= h_in, phi_2: x >= h_out
            pivot = req[1]    #curr location
            d1 = req[2][0]      #distance min
            d2 = req[2][1]      #distance max
            h_in = req[3]
            h_out = req[4]
                        
#            s = calculateDNF(('mu', pivot, h_in), t, True, T, NV)   #calculate pivot >= h_in           
            s = calculateDNF(('mu', pivot, h_in), t, False, T, NV)   #calculate pivot <= h_in
            for node in range(0, NV):
                if adj[node, pivot] < d1 and node!= pivot:   #specify node in phi
#                    set_in_curr = calculateDNF(('mu', node, h_in), t, True, T, NV)     #>= h_in is the lower bound                   
                    set_in_curr = calculateDNF(('mu', node, h_in), t, False, T, NV)     #<= h_in is the upper bound
                    #s_in = torch.cat((s_in, set_in_curr), dim=0)
                    sc = torch.zeros((s.shape[0], set_in_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s[:, :, :, 0].unsqueeze(1), set_in_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s[:, :, :, 1].unsqueeze(1), set_in_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s = sc
                    
                elif adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
#                    set_out_curr = calculateDNF(('mu', node, h_out), t, False, T, NV)    #<= h_out is the lower bound
                    set_out_curr = calculateDNF(('mu', node, h_out), t, True, T, NV)    #>= h_out is the upper bound
                    #s_out = torch.cat((s_out, set_out_curr), dim=0)
                    sc = torch.zeros((s.shape[0], set_out_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s[:, :, :, 0].unsqueeze(1), set_out_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s[:, :, :, 1].unsqueeze(1), set_out_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s = sc
            
            return simplifyDNF(s)
            #return s
            
        elif req[0] == "somewhere":     #if current value is high, there should not exist another high value that is close to it
            
            pivot = req[1]    #curr location
            d1 = req[2][0]      #distance min
            d2 = req[2][1]      #distance max
            h = req[3]           
            s = calculateDNF(('mu', pivot, h), t, False, T, NV)   #calculate pivot, <= h

            for node in range(0, NV):                    
                if adj[node, pivot] >= d1 and adj[node, pivot] <= d2:   
                    set_out_curr = calculateDNF(('mu', node, h), t, False, T, NV)    #nearby value cant be too high, <= h
                    sc = torch.zeros((s.shape[0], set_out_curr.shape[0], T, NV, 2))     #merge
                    sc[:, :, :, :, 0] = torch.max(s[:, :, :, 0].unsqueeze(1), set_out_curr[:, :, :, 0].unsqueeze(0))
                    sc[:, :, :, :, 1] = torch.min(s[:, :, :, 1].unsqueeze(1), set_out_curr[:, :, :, 1].unsqueeze(0))
                    sc = sc.view(-1, T, NV, 2)                    
                    s = sc
            
            return simplifyDNF(s)

            
def trace_gen(DNF_array, trace):
    dist1 = DNF_array[:, :, :, 0].unsqueeze(0) - trace.unsqueeze(1)     #difference of DNF min and trace
    dist2 = trace.unsqueeze(1) - DNF_array[:, :, :, 1].unsqueeze(0)     #difference of DNF max and trace
    
    dist1 = torch.max(dist1, torch.zeros(dist1.size()))     #max(dist, 0)
    dist2 = torch.max(dist2, torch.zeros(dist2.size()))
    dnf_select = torch.max(dist1, dist2).sum(dim=2).sum(dim=2).argmin(dim=1)
    selected_dnf = DNF_array[dnf_select]
    trace_stl = torch.max(trace, selected_dnf[:, :, :, 0])      #max of minimum value
    trace_stl = torch.min(trace_stl, selected_dnf[:, :, :, 1])      #min of maximum value
    return trace_stl
    
if __name__=='__main__':
    mu_req = ('mu', 0, 5)
    always_req = ('always', (0, 5), ('mu', 0, 5))
    eventually_req = ('eventually', (0, 5), ('mu', 0, 5))
    always_even_req = ('always', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
    even_even_req = ('eventually', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
    mu_dnf = calculateDNF(mu_req, 0, True, 5)
    #print(calculateDNF(mu_req, 0, True, 5))
    #print(calculateDNF(always_req, 0, True, 6))
    #print(calculateDNF(eventually_req, 0, True, 6))
    even_dnf = calculateDNF(even_even_req, 0, True, 11)
    #print(even_dnf)
    #print(simplifyDNF(even_dnf))
    
    mu_req_2 = ('mu', 1, 5)
    always_req_2 = ('always', (0, 5), ('mu', 0, 5))
    eventually_req_2 = ('eventually', (0, 5), ('mu', 1, 5))
    always_even_req_2 = ('always', (0, 5), ('eventually', (0, 5), ('mu', 0, 5)))
    even_even_req_2 = ('eventually', (0, 5), ('eventually', (0, 5), ('mu', 1, 5)))
    #print(calculateDNF(mu_req_2, 0, True, 5, 2))
    #print(calculateDNF(always_req_2, 0, True, 6, 2))
    #print(calculateDNF(eventually_req_2, 0, True, 6, 2))
    even_dnf_2 = calculateDNF(even_even_req_2, 0, True, 11, 2)
    #print(even_dnf_2)
    temp = simplifyDNF(even_dnf_2).numpy()[:, :, :, 0]  #(11,11,2,2)
    #print(simplifyDNF(even_dnf_2))
    #print(adj)
    surround_req = ("surround", 0, (0.1, 0.5), 60, 40)      #location is 1 
    dnf = calculateDNF(surround_req, 3, True, 10, 6)   #(req, t, flag, T=10, NV=4) (1,10,4,2)
    #temp_np = dnf.numpy()[0, :, :, :]
    #print(dnf)
    
def gen_train_stl(data, timeunites, pastunites):
    if data=='step':
        return ("and", ("always", (0,8), ("neg", ("mu", 0, 1.00))), ("always", (14, timeunites-1), ("mu", 0, 1.00)))
    elif data=='multistep':
        return ('and', ("and", ("always", (0,8), ("neg", ("mu", 0, 1.00))),
         ("always", (14, timeunites-1), ("mu", 0, 1.00))), ("and", ("always", (0,8), ("neg", ("mu", 1, 1.00))), ("always", (14, timeunites-1), ("mu", 1, 1.00))))   
    elif data=='cont':
        return ("always", (0,timeunites-5), ("and", ("eventually", (0,4), ("mu", 0, 0)),("eventually", (0,4), ("neg", ("mu", 0, 0)))))
    elif data=='multicont':
        return ("always", (0,timeunites-5), ("and", ("and", ("eventually", (0,4), ("mu", 0, 0)),
            ("eventually", (0,4), ("neg", ("mu", 0, 0)))), ("and", ("eventually", (0,4), ("mu", 1, 0)),("eventually", (0,4), ("neg", ("mu", 1, 0))))))
    elif data=='even':
        return ("eventually", (0, timeunites-1), ("mu", 0, 0.99))
    elif data=='multieven':
        return ('and', ("eventually", (0, timeunites-1), ("mu", 0, 0.99)), ("eventually", (0, timeunites-1), ("mu", 1, 0.99)))
    elif data=='jump':
        return ("and", ("always", (0, timeunites-1), ("neg", ("mu", 0, 1.0))), ("always", (0, timeunites-1), ("mu", 0, -1.0)))
    elif data=='air':
        return ("and", ("eventually", (0, timeunites-1), ("mu", 0, 0.0)), ("eventually", (0, timeunites-1), ("neg", ("mu", 0, 0.0))))
    elif data=='multi':
        return ("always", (0, timeunites-1), ("or", ("and", ("mu", 0, 1.0), ("mu", 1, 0.5)), ("and", ("neg", ("mu", 0, 1.0)), ("neg", ("mu", 1, 0.5)))))
    elif data=='multijump':
        return ("always", (0, timeunites-1), ("and", ("and", ("mu", 0, -1.0), ("neg", ("mu", 0, 1.0))), ("and", ("mu", 1, -1.0), ("neg", ("mu", 1, 1.0)))))
    elif data=='traffic1':
        return ("and", ("always", (0, timeunites-1), ("mu", 0, 0)), ("always", (0, timeunites-1), ("neg", ("mu", 0, 30.0))))
    elif data=='traffic2':
        return ("and", ("and", ("always", (0, 8), ("neg", ("mu", 0, 60.0))), ("always", (9,12), ("mu", 0, 40.0))), ("always", (13, timeunites-1), ("neg", ("mu", 0, 60.0))))
    elif data=='airmulti':
        #return ("always", (0, 4), ("mu", 1, 7))    #68.1-->68.1     7.2 --> 7.2
        #return ("and", ("always", (0, timeunites-1), ("mu", 0, 0)), ("always", (0, timeunites-1), ("neg", ("mu", 0, 30.0))))    #1.8 -->28,18
        #return ("eventually", (0, timeunites-1), ("mu", 0, 0.99))   #100 --> 100
        #return ("and", ("always", (0,8), ("neg", ("mu", 0, 1.00))), ("always", (14, timeunites-1), ("mu", 0, 1.00))) #0 --> 8%
        #return ("always", (0, timeunites-1), ("and", ("neg", ("mu", 0, 100)),  ("neg", ("mu", 1, 100))))        #teacher minor improvement, not 100%
        return gen_airmulti_stl(timeunites)
    elif data=='unusual':
        return ("always", (0, 4), ("or", ("neg", ("mu", 0, 499.99)), ("always", (1, 9), ("mu", 1, 9))))
    elif data=='consecutive':
        return ("always", (0, timeunites-1), ("and", ("neg", ("mu", 0, 100)),  ("neg", ("mu", 1, 100))))      
    elif data=='airpde1':
        return gen_airpde1_stl(timeunites, nvar=35, d1=0.05, d2=0.2, hin=50, hout=0)             
    elif data=='airpde2':
        return gen_airpde2_stl(timeunites, nvar=35, d1=0, d2=0.1, h=50)    
        
        
def gen_airmulti_stl(timeunites, nvar=36, th=80):
    #print(nvar)
    base0 = ("mu", 0, 0)
    base = ("and", ("mu", 1, -th), ("neg", ("mu", 1, th)))
    base = ("and", base0, base)
    for i in range(2, nvar):
        base = ("and", base, ("and", ("mu", i, -th), ("neg", ("mu", i, th))))   # -th <= loc_i <= th
    base = ("always", (5, timeunites-1), base)
    return base
    
def gen_airpde1_stl(timeunites, nvar=35, d1=0.05, d2=0.2, hin=50, hout=30):
    #base0 = ("neg", ("surround", 0, (d1, d2), hin, hout))
    #base = ("neg", ("surround", 1, (d1, d2), hin, hout))
    base0 = (("surround", 0, (d1, d2), hin, hout))
    base = ( ("surround", 1, (d1, d2), hin, hout))
    base = ("and", base0, base)
    
    for i in range(2, int(nvar)):
        #base = ("and", base, ("neg",("surround", i, (d1, d2), hin, hout)))
        base = ("and", base, (("surround", i, (d1, d2), hin, hout)))
    base = ("always", (5, timeunites - 1), base)  #timeunites = 24
    return base    
    
def gen_airpde2_stl(timeunites, nvar=35, d1=0, d2=0.1, h=40):
    base0 = ("and", ("somewhere", 0, (d1, d2), h), ("mu", 0, 0))
    base = ("and", ("somewhere", 1, (d1, d2), h), ("mu", 1, 0))
    base = ("and", base0, base)
    for i in range(2, int(nvar)):
        base = ("and", base, ("and",("somewhere", i, (d1, d2), h), ("mu", i, 0)))
    base = ("always", (5, timeunites-1), base)
    return base     
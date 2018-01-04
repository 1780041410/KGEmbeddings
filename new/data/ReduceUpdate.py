
# coding: utf-8

# In[1]:

import os
import numpy as np
import sys


# In[9]:

class HandleFile(object):
    def __init__(self,filename,Newfilename):
        self.file = filename
        self.Newfile = Newfilename
    def Read(self):
        file = open(self.file,'r')
        PosTriple = []
        NegTriple = []
        for item in file:
            item = item.strip().split('\t')
            if(item[3] == '1'):
                PosTriple.append(item)
            else:
                NegTriple.append(item)
                
        file.close()
        
        return  PosTriple,NegTriple
    
    def ReduceUpdate(self,NegTriple,PosTriple,num):
        self.neg = NegTriple
        self.pos = PosTriple
        NewTriple = list()
        NegSize = len(NegTriple)
        PosSize = len(PosTriple)
        
        negnum = int(num * 0.1)
        posnum = int(num * 0.9)
        
        for i in range(posnum):
            item = np.random.randint(PosSize)
            if PosTriple[item] not in NewTriple:
                NewTriple.append(PosTriple[item])
        
        for i in range(negnum):
            item = np.random.randint(NegSize)
            if NegTriple[item] not in NewTriple:
                NewTriple.append(NegTriple[item])
        
        return NewTriple
        
        
    
    def WriteFile(self,NewTriple):
        file = open(self.Newfile,'w')
        
        for item in NewTriple:
            file.write(item[0]+'\t'+item[1]+'\t'+item[2]+'\t'+item[3]+'\n')
    
        file.close()


# In[10]:

def init():
    filename = str(sys.argv[1])
    Newfilename = str(sys.argv[2])
    num = str(sys.argv[3])
    num = int(num)
    
    WN18 = HandleFile(str('./WN18/'+filename),str('./WN18/'+Newfilename))
   
    wn18PosTriple,wn18NegTriple = WN18.Read()
    wn18NewTriple = WN18.ReduceUpdate(wn18PosTriple,wn18NegTriple,num)
    WN18.WriteFile(wn18NewTriple)

    WN11 = HandleFile('./WN11/'+filename,'./WN11/'+Newfilename)
    wn11PosTriple,wn11NegTriple = WN11.Read()
    wn11NewTriple = WN11.ReduceUpdate(wn11PosTriple,wn11NegTriple,num)
    WN11.WriteFile(wn11NewTriple)

    
    
    FB13 = HandleFile('./FB13/'+filename,'./FB13/'+Newfilename)
    fb13PosTriple,fb13NegTriple = FB13.Read()
    fb13NewTriple = FB13.ReduceUpdate(fb13,PosTriple,fb13NegTriple,num)
    FB13.WriteFile(fb13NewTriple)

    FB15k = HandleFile('./FB15k/'+filename,'./FB15k/'+Newfilename,)
    fb15kPosTriple,fb15kNegTriple = FB15k.Read()
    fb15kNewTriple = FB15k.ReduceUpdate(fb15kPosTriple,fb15kNegTriple,num)
    FB15k.WriteFile(fb15kNewTriple)


# In[11]:

if __name__ == '__main__':
    init()


# In[ ]:




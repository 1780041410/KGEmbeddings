
# coding: utf-8

# In[1]:

import os
import numpy as np
import sys


# In[2]:

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
    
    def ReduceNeg(self,NegTriple):
        num = len(NegTriple)
        self.neg = NegTriple
        NewTriple = list()
        for i in range(int(num/2)):
            item = np.random.randint(num)
            if NegTriple[item] not in NewTriple:
                NewTriple.append(NegTriple[item])
        
        return NewTriple
    
    def WriteFile(self,PosTriple,NewTriple):
        file = open(self.Newfile,'w')
        self.pos = PosTriple
        self.Newneg = NewTriple
        for item in PosTriple:
            file.write(item[0]+'\t'+item[1]+'\t'+item[2]+'\t'+item[3]+'\n')
        
        for item in NewTriple:
            file.write(item[0]+'\t'+item[1]+'\t'+item[2]+'\t'+item[3]+'\n')
    
        file.close()
        
    def WriteLog(self,logfile):
        file = open(logfile,'a')
        file.write(self.file+'\t'+'PosTriples'+'\t'+ str(len(self.pos))+'NegTriple'+'\t'+str(len(self.neg))+'\n')
        file.write(self.Newfile+'\t'+'PosTriples'+'\t'+ str(len(self.pos))+'NegTriple'+'\t'+str(len(self.Newneg))+'\n')
        
        
        


# In[3]:

def init():
    filename = str(sys.argv[1])
    Newfilename = str(sys.argv[2])
    logfile = 'Reducelog.txt'
    
    WN18 = HandleFile(str('./WN18/'+filename),str('./WN18/'+Newfilename))
   
    wn18PosTriple,wn18NegTriple = WN18.Read()
    wn18NewTriple = WN18.ReduceNeg(wn18NegTriple)
    WN18.WriteFile(wn18PosTriple,wn18NewTriple)
    WN18.WriteLog('./WN18/'+logfile)
    
    WN11 = HandleFile('./WN11/'+filename,'./WN11/'+Newfilename)
    wn11PosTriple,wn11NegTriple = WN11.Read()
    wn11NewTriple = WN11.ReduceNeg(wn11NegTriple)
    WN11.WriteFile(wn11PosTriple,wn11NewTriple)
    WN11.WriteLog('./WN11/'+logfile)
    
    
    FB13 = HandleFile('./FB13/'+filename,'./FB13/'+Newfilename)
    fb13PosTriple,fb13NegTriple = FB13.Read()
    fb13NewTriple = FB13.ReduceNeg(fb13NegTriple)
    FB13.WriteFile(fb13PosTriple,fb13NewTriple)
    FB13.WriteLog('./FB13/'+logfile)
    
    FB15k = HandleFile('./FB15k/'+filename,'./FB15k/'+Newfilename,)
    fb15kPosTriple,fb15kNegTriple = FB15k.Read()
    fb15kNewTriple = FB15k.ReduceNeg(fb15kNegTriple)
    FB15k.WriteFile(fb15kPosTriple,fb15kNewTriple)
    FB15k.WriteLog('./FB15k/'+logfile)


# In[4]:

if __name__ == "__main__":
    
    init()


# In[ ]:




# In[ ]:




# In[ ]:




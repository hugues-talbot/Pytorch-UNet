
# coding: utf-8

# ## Prediction
# 
# Using the prediction code is kind of not simple

# In[29]:


from os import listdir
from os.path import isfile, join
import subprocess
import shlex
import re


# In[11]:


onlyfiles = [f for f in listdir("data/imgs") if isfile(join("data/imgs", f))]


# In[80]:


s=""
o=""
imgpath="data/imgs/"
i=0
for file in onlyfiles:
    bn=re.sub(".png","",file)
    s+="data/imgs/"+bn+".png "
    o+="data/predict/"+bn+".gif "

mymodel="MODEL_baseline.pth"
cmd="/home/talboth/anaconda3/envs/radar/bin/python ./predict.py --model models/"+mymodel+" --input "+s+" --output "+o


# In[77]:


c=shlex.split(cmd)


# In[78]:


subprocess.run(c,stderr=subprocess.STDOUT)


import numpy as np

#c_71 = np.genfromtxt('submission-3.csv',delimiter=",",skip_header=0)
#v,c = np.unique(c_71[:,1],return_counts=True)
#print("perfomance of the 71 model {},{}".format(v,c))
#
#c_71 = np.genfromtxt('submission-2.csv',delimiter=",",skip_header=0)
#v,c = np.unique(c_71[:,1],return_counts=True)
#print("perfomance of the 33 model {},{}".format(v,c))
#
#c_71 = np.genfromtxt('submission-1.csv',delimiter=",",skip_header=0)
#v,c = np.unique(c_71[:,1],return_counts=True)
#print("perfomance of the 17 model {},{}".format(v,c))

c_71 = np.genfromtxt('submission.csv',delimiter=",",skip_header=0)
v,c = np.unique(c_71[:,1],return_counts=True)
print("perfomance of the current model {},{}".format(v,c))

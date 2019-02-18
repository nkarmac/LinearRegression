# solves normal equation to learn linear regression models

import sys
import numpy as np 


def main():

    i = 0       # line num for iterating
    n = 0       # num data pts
    d = 0       # num features
    w_vec = 0   # weight vector
    y_vec = 0   # label vector
    x_vec = 0   # feature vector

    myfile = input("Please specify a data file: (eg \"data_10k_100.tsv\")\n")
    datafile = open(myfile)
    for line in datafile:
        
        if i==0:    # number of data pts
            n = int(line)
            i += 1
            continue
        if i==1:    # number of features
            d = int(line)
            y_vec = np.empty((n,1))
            x_vec = np.empty((n,d))
            bias = np.ones((n,1))
            i += 1
            continue
        if i==2:    # skipping label lines
            i += 1
            continue

       # build vectors 
        mylist = np.fromstring(line, dtype=np.float, sep='\t')
        y_vec[i-3] = [mylist[0]]
        x_vec[i-3] = mylist[1:]
        
        i += 1
    
    # add bias to x_vec
    x_vec = np.append(x_vec,bias, axis=1)

    # weight vector using normal equation
    w_vec = np.linalg.inv((np.transpose(x_vec) @ x_vec)) @ (np.transpose(x_vec) @ y_vec)

    for x in range(d+1):
        print("w%i" % (d-x), end='\t')
    print()
    
    for x in range(d+1):
        print("%f" % w_vec[x], end='\t' )


if __name__ == "__main__":
    main()
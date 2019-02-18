# solves normal equation to learn linear regression models

import numpy as np 


def main():

    i = 0       # line num for iterating
    n = 0       # num data pts
    d = 0       # num features
    w_vec = 0   # weight vector
    y_vec = 0   # label vector
    x_vec = 0   # feature vector
    loss_vec = 0
    #inner_vec = 0

    learn_rate = np.float((input("Please specify a learn rate: (eg \"0.000001\")\n")))
    epochs = int((input("Please specify the number of epochs: (eg \"200\")\n")))
    
    myfile = input("Please specify a data file: (eg \"data_10k_100.tsv\")\n")
    datafile = open(myfile)
    for line in datafile:
        
        if i==0:    # number of data pts
            n = int(line)
            i += 1
            continue
        if i==1:    # number of features
            d = int(line)

            # initialze vectors
            y_vec = np.empty((n,1))
            x_vec = np.empty((n,d))
            bias = np.ones((n,1))
            loss_vec = np.empty((n,1))
            #inner_vec = np.empty((n,1))
            w_vec = np.random.random_sample((d+1,1))

            i += 1
            continue
        if i==2:    # skipping label line
            i += 1
            continue

       # build vectors 
        mylist = np.fromstring(line, dtype=np.float, sep='\t')
        y_vec[i-3] = [mylist[0]]
        x_vec[i-3] = mylist[1:]
        
        i += 1
    
    # add bias to x_vec
    x_vec = np.append(x_vec,bias, axis=1)

    for i in range(epochs):
        w_vec = w_vec - (learn_rate / n) * (np.transpose(x_vec) @ x_vec @ w_vec - (np.transpose(x_vec) @ y_vec))

        
        
        ### This way also works, but is many times slower
        # for j in range(d):
        #     for x in range(n):
        #         inner_vec[x] = (y_vec[x] - (np.transpose(w_vec) @ x_vec[x])) * x_vec[x][j]

        #     w_vec[j] = w_vec[j] + (learn_rate / n) * np.sum(inner_vec)
    
    # Build loss function
    for x in range(n):
        loss_vec[x] = np.square(y_vec[x] - np.transpose(w_vec) @ x_vec[x])
    
    # Sum then average loss for each iteration
    AverageLoss = np.sum(loss_vec) / (2*n)
    print("\nAverage Loss is: %f" % AverageLoss)


if __name__ == "__main__":
    main()
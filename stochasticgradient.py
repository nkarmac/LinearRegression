# uses stochastic gradient descent to learn linear regression models

import numpy as np 


def main():

    i = 0       # line num for iterating
    n = 0       # num data pts
    m = 1       # batch size
    d = 0       # num features
    w_vec = 0   # weight vector
    y_vec = 0   # label vector
    x_vec = 0   # feature vector
    loss_vec = 0
    batch_size = 1
    inner_vec = 0

    learn_rate = np.float(input("Please specify a learning rate: (eg \"0.000001\")\n"))
    epochs = int(input("Please specify the number of epochs: (eg \"20\")\n"))
    
    myfile = input("Please specify a data file: (eg \"data_10k_100.tsv\")\n")
    print("opening file...")
    datafile = open(myfile)
    print("building dataset...")
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
            bias = np.ones((n,1), dtype=np.float64)
            loss_vec = np.empty((n,1))
            inner_vec = np.empty((n,1))
            w_vec = np.random.random_sample((d+1,1))

            i += 1
            continue
        if i==2:    # skipping label line
            i += 1
            continue

       # build vectors 
        mylist = np.fromstring(line, dtype=np.float64, sep='\t')
        y_vec[i-3] = [mylist[0]]
        x_vec[i-3] = mylist[1:]
        
        i += 1
    
    # add bias to x_vec
    x_vec = np.append(x_vec,bias, axis=1)

    print("updating weights...")
    for epoch in range(epochs):
        print("Epoch %i..." % (epoch+1))
        randombatch = np.split(x_vec, n)

        i = 0
        for x in (randombatch):
            yhat = np.dot(np.transpose(w_vec),x[0])
            loss = y_vec[i] - yhat
            for j in range(d):
                w_vec[j] = w_vec[j] + (learn_rate) * loss * x[0][j]
            i+=1
        
    # ### Printing Block, uncomment for w output
    # for x in range(1, d+1):
    #     print("w%i" % (i), end='\t')
    # print()
    
    # for x in range(d+1):
    #     print("%f" % w_vec[x], end='\t' )
    # ###


    # Build loss function
    for x in range(n):
        loss_vec[x] = np.square(y_vec[x] - np.transpose(w_vec) @ x_vec[x])

    # Sum then average loss - ***Too slow to show during iterations***
    AverageLoss = np.sum(loss_vec) / (2*n)
    print("Average Loss is: %f" % AverageLoss)


if __name__ == "__main__":
    main()
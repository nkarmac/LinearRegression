# uses batch gradient descent to learn linear regression models

import numpy as np 


def main():

    i = 0       # line num for iterating
    n = 0       # num data pts
    d = 0       # num features
    w_vec = 0   # weight vector
    y_vec = 0   # label vector
    x_vec = 0   # feature vector
    loss_vec = 0

    learn_rate = np.float(0.000001)
    epochs = 200
    
    myfile = input("Please specify a data file: (eg \"data_10k_100.tsv\")\n")
    print("opening file...")
    datafile = open(myfile)
    "building dataset vectors..."
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

    print("updating weights...")
    for i in range(epochs):
        loss = np.dot(x_vec,w_vec) - y_vec
        w_vec = w_vec - (learn_rate / n) * np.dot(np.transpose(x_vec), loss)
    

    # ### Printing Block, uncomment for w output
    # for x in range(d+1):
    #     print("w%i" % (d-x), end='\t')
    # print()
    
    # for x in range(d+1):
    #     print("%f" % w_vec[x], end='\t' )
    # ###

    # Build loss function
    for x in range(n):
        loss_vec[x] = np.square(y_vec[x] - np.transpose(w_vec) @ x_vec[x])
    
    # Sum then average loss for each iteration
    AverageLoss = np.sum(loss_vec) / (2*n)
    print("Average Loss is: %f" % AverageLoss)

if __name__ == "__main__":
    main()
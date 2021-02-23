import numpy as np





#This returns an array with first column the location of the start of the bin, the second column as the density (number of cars)
#in that bin and the third as the average velocity of the cars in that bin.
#NOTE the last entry for column 2 and 3 is always zero to allow us to define intervals well.
def indicator_func(x,v,bin_size,n_bins):
    bin_matrix=np.zeros([3,n_bins+1])
    bin_matrix[0]=-np.arange(n_bins+1)*bin_size
    i_car=0
    while i_car<len(x):
        if x[i_car]==-np.inf:
            x=np.delete(x,i_car)
            v=np.delete(v,i_car)
        else:
            i_car += 1
    ind=np.argsort(x)[::-1]
    bin_indicator=np.ceil(x[ind]/bin_size)
    v_sorted=v[ind]
    bin_end=np.zeros(n_bins+1,dtype=int)
    for i_bin in range(1,n_bins+1):
        bin_end[i_bin]=bin_end[i_bin-1]
        if -i_bin+1 in bin_indicator:
            bin_end[i_bin]=np.where(bin_indicator==-i_bin+1)[0][-1]
            bin_matrix[1,i_bin-1]=(bin_end[i_bin]-bin_end[i_bin-1]+(i_bin==1))/len(x)
            bin_matrix[2,i_bin-1]=np.mean(v_sorted[bin_end[i_bin-1]+(i_bin>1):bin_end[i_bin]+1])
    return bin_matrix


def abs_weighted_indicator_func(x, v, bin_size, n_bins):
    bin_matrix = np.zeros([3, n_bins + 1])
    bin_matrix[0] = -np.arange(n_bins + 1) * bin_size
    i_car = 0
    while i_car < len(x):
        if x[i_car] == -np.inf:
            x = np.delete(x, i_car)
            v = np.delete(v, i_car)
        else:
            i_car += 1
    ind = np.argsort(x)[::-1]
    x_sorted = x[ind]
    bin_indicator = np.ceil(x_sorted[x_sorted >= -bin_size * n_bins] / bin_size - .5)
    v_sorted = v[ind]
    bin_end = np.zeros(n_bins + 2, dtype=int)

    for i_bin in range(1, n_bins + 2):
        bin_end[i_bin] = bin_end[i_bin - 1]
        if -i_bin + 1 in bin_indicator:
            bin_end[i_bin] = np.where(bin_indicator == -i_bin + 1)[0][-1]
        if i_bin == 1:
            bin_matrix[1, i_bin - 1] = (bin_end[i_bin] - bin_end[i_bin - 1] + (i_bin == 1)) / len(x)
            bin_matrix[2, i_bin - 1] = np.sum(v_sorted[bin_end[i_bin - 1] + (i_bin > 1):bin_end[i_bin] + 1]) / len(x)
        elif i_bin == n_bins + 1:
            bin_matrix[1, i_bin - 2] += (bin_end[i_bin] - bin_end[i_bin - 1] + (i_bin == 1)) / len(x)
            bin_matrix[2, i_bin - 2] += np.sum(v_sorted[bin_end[i_bin - 1] + (i_bin > 1):bin_end[i_bin] + 1]) / len(x)
        else:
            #             print(i_bin)
            for i_car in range(bin_end[i_bin - 1] + 1, bin_end[i_bin] + 1):
                #                 print(i_car)
                weight = abs(x_sorted[i_car] - (-i_bin + 1.5) * bin_size) / bin_size
                #                 print(weight)
                bin_matrix[1, i_bin - 2] += (1 - weight) / len(x)
                bin_matrix[1, i_bin - 1] += (weight) / len(x)
                bin_matrix[2, i_bin - 2] += v_sorted[i_car] * (1 - weight) / len(x)
                bin_matrix[2, i_bin - 1] += v_sorted[i_car] * (weight) / len(x)
    #     print(bin_end)
    return bin_matrix
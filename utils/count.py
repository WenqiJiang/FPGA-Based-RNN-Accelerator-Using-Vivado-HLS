# f = open("./dset.asci", "r")
# f.split(",")

with open("./dense_1_bias.txt", 'r') as myfile:
    data = myfile.read().split(',')
    count = len(data)
    print(count)
    for i in range(10):
        print(data[count - i - 1])

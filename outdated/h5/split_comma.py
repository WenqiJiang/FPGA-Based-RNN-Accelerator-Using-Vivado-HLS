with open("./dense_1_bias.txt", 'r') as myfile:
    data = myfile.read().split(",")

print(data)

with open("./dense_1_bias.txt", 'w') as myfile:
    for elements in data:
        myfile.write("%s" % elements)
    #myfile.writelines(["%s " % floats for floats in data])
    print(data)

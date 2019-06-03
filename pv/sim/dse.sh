# DSE (in the future)
# for W in [8,32]
#   for I in [0, W]
#     check accuracy and
#     keep track of which couple (W,I) with the smalles W gives us the best accuracy

echo "INFO: clean the working directory"
make clean

echo "INFO: compile"
FXD_W_LENGTH=32 FXD_I_LENGTH=16 V=1 make

echo "INFO: run"
if [ $? == 0 ]; then
    ./c-rnn
fi

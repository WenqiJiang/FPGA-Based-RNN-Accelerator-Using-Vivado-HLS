# DSE (in the future)
# for W in [8,32]
#   for I in [0, W]
#     check accuracy and
#     keep track of which couple (W,I) with the smalles W gives us the best accuracy

# a sample test
START_LEN=8
END_LEN=11
LOG="dse_log_${START_LEN}_${END_LEN}.log"
touch $LOG

for W in $(seq $START_LEN $END_LEN)
do
	for I in $(seq 0 $W)
	do
		echo "W: $W	I: $I"
		echo "INFO: clean the working directory"
		make clean

		echo "INFO: compile"
		FXD_W_LENGTH=$W FXD_I_LENGTH=$I V=1 make

	echo "INFO: run"
	if [ $? == 0 ]; then
		echo "W: $W	I: $I" >> $LOG
	    	./c-rnn >> $LOG
	else
		echo "compilation FAILED! W: $W I:$I"
		exit 1
	fi
	done
done

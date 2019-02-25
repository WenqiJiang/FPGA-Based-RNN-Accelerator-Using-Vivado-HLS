# We’ll use gcc for C compilation and g++ for C++ compilation
# CC = gcc
# CXX = g++
# Let’s leave a place holder for additional include directories
# INCLUDES =
# Compilation options:
# -g for debugging info and -Wall enables all warnings
# CFLAGS = -g -Wall $(INCLUDES)
# CXXFLAGS = -g -Wall $(INCLUDES)
# Linking options:
# -g for debugging info
# LDFLAGS = -g
# List the libraries you need to link with in LDLIBS
# For example, use "-lm" for the math library
# LDLIBS =

main: main.o params_init.o rnn.o fc.o

main.o: main.c 
params_init.o: params_init.c
rnn.o: rnn.c
fc.o: fc.c

clean:
	rm *.o main


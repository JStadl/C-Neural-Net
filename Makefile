CC      ?= gcc
CFLAGS  ?= -std=c17 -g\
	-D_POSIX_SOURCE -D_DEFAULT_SOURCE\
	-Wall -pedantic
all: main

main: main.o linalg.o ai.o
	$(CC) $(LDFLAGS) main.o linalg.o ai.o -o main -lm

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

linalg.o: linalg.c linalg.h
	$(CC) $(CFLAGS) -c linalg.c -o linalg.o

ai.o: ai.c ai.h
	$(CC) $(CFLAGS) -c ai.c -o ai.o

# remove object files and executable when user executes "make clean"
clean:
	rm *.o main

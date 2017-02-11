CC=gcc
CPPFLAGS=-I/usr/local/include/eigen3

all: sndrone

sndrone: sndrone_main.o
	g++ -Wall sndrone_main.o -o sndrone

clean:
	rm -f sndrone_main.o sndrone

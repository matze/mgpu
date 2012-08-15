CFLAGS=`pkg-config --cflags glib-2.0 gthread-2.0` --std=c99 -Wall
LDFLAGS=`pkg-config --libs glib-2.0 gthread-2.0`
OCLFLAGS=-I/usr/local/cuda/include

all: mgpu

mgpu: main.c
	$(CC) $(CFLAGS) $(OCLFLAGS) -O3 -o mgpu main.c $(LDFLAGS) -lOpenCL

clean:
	rm -f mgpu

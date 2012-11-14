CFLAGS=$(shell pkg-config --cflags glib-2.0 gthread-2.0) --std=c99 -Wall -Werror
LDFLAGS=$(shell pkg-config --libs glib-2.0 gthread-2.0)
OCLFLAGS=-I/usr/local/cuda/include
SOURCES=main.c ocl.c

all: mgpu

mgpu: $(SOURCES)
	$(CC) $(CFLAGS) $(OCLFLAGS) -O3 -o mgpu $(SOURCES) $(LDFLAGS) -lOpenCL

clean:
	rm -f mgpu

all: mgpu

mgpu: main.c
	gcc -I/usr/local/cuda/include `pkg-config --libs --cflags glib-2.0 gthread-2.0` -lOpenCL --std=c99 -O3 -o mgpu main.c


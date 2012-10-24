# mgpu

`mgpu` is a small benchmark program to check the concurrent performance of
multiple GPUs.

In order to compile with `make` you might have to adjust the `OCLFLAGS` variable
in the Makefile that must point to you OpenCL source directory.

You can also dump the performance values of all OpenCL events, by setting
`do_profile` to TRUE in `main.c`. The data is output on `stdout`.

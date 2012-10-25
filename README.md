# mgpu

`mgpu` is a small benchmark program to check the concurrent performance of
multiple GPUs.

In order to compile with `make` you might have to adjust the `OCLFLAGS` variable
in the Makefile that must point to you OpenCL source directory.

## Screenshot

The help screen

    $ ./mgpu --help
    Usage:
      mgpu [OPTION...]  - test multi GPU performance

    Help Options:
      -?, --help             Show help options

    Application Options:
      -n, --num-images=N     Number of images
      -w, --width=W          Width of imags
      -h, --height=H         Height of images
      --enable-profiling     Enable profiling

`mgpu` in action

    $ ./mgpu -n 12
    # Platform: OpenCL 1.1 CUDA 4.2.1
    # Device 0: GeForce GTX 580
    # Device 1: GeForce GTX 580
    # Computing <nlm> for 12 images of size 1024x1024
    # Single GPU: total = 7.021065s, time per image = 0.585089s, error = 0.000000
    # Single Threaded, Multi GPU: total = 6.437301s, time per image = 0.536442s, error = 0.000000
    # Multi Threaded, Multi GPU: total = 3.508974s, time per image = 0.292415s, error = 0.000000


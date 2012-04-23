
#include <CL/cl.h>
#include <glib-2.0/glib.h>
#include <stdio.h>

static const gboolean DO_PROFILE = FALSE;

typedef struct {
    cl_context context;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_command_queue *cmd_queues;

    GList *kernel_table;
    GHashTable *kernels;         /**< maps from kernel string to cl_kernel */
} opencl_desc;

typedef struct {
    guint num_images;
    guint width;
    guint height;
    gsize image_size;
    gfloat **host_data;
    gfloat **single_result;
    gfloat **multi_result;
    cl_mem *dev_data_in;
    cl_mem *dev_data_out;
    cl_event *events;
    cl_event *read_events;
} Benchmark;

typedef struct {
    Benchmark *benchmark;
    opencl_desc *ocl;
    cl_kernel *kernels;
    guint batch_size;
    guint thread_id;
} ThreadLocalBenchmark;

static const gchar* opencl_error_msgs[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",

    /* next IDs start at 30! */
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
};

/**
 * \brief Returns the error constant as a string
 * \param[in] error A valid OpenCL constant
 * \return A string containing a human-readable constant or NULL if error is
 *      invalid
 */
const gchar* opencl_map_error(int error)
{
    if (error >= -14)
        return opencl_error_msgs[-error];
    if (error <= -30)
        return opencl_error_msgs[-error-15];
    return NULL;
}

#define CHECK_ERROR(error) { \
    if ((error) != CL_SUCCESS) g_message("OpenCL error <%s:%i>: %s", __FILE__, __LINE__, opencl_map_error((error))); }

static gchar *ocl_read_program(const gchar *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
        return NULL;

    fseek(fp, 0, SEEK_END);
    const size_t length = ftell(fp);
    rewind(fp);

    gchar *buffer = (gchar *) g_malloc0(length+1);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    size_t buffer_length = fread(buffer, 1, length, fp);
    fclose(fp);
    if (buffer_length != length) {
        g_free(buffer);
        return NULL;
    }
    return buffer;
}

cl_program ocl_get_program(opencl_desc *ocl, const gchar *filename, const gchar *options)
{
    gchar *buffer = ocl_read_program(filename);
    if (buffer == NULL) 
        return FALSE;

    int errcode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char **) &buffer, NULL, &errcode);

    if (errcode != CL_SUCCESS) {
        g_free(buffer);
        return NULL;
    }

    errcode = clBuildProgram(program, ocl->num_devices, ocl->devices, options, NULL, NULL);

    if (errcode != CL_SUCCESS) {
        const int LOG_SIZE = 4096;
        gchar* log = (gchar *) g_malloc0(LOG_SIZE * sizeof(char));
        CHECK_ERROR(clGetProgramBuildInfo(program, ocl->devices[0], CL_PROGRAM_BUILD_LOG, LOG_SIZE, (void*) log, NULL));
        g_print("\n=== Build log for %s===%s\n\n", filename, log);
        g_free(log);
        g_free(buffer);
        return NULL;
    }

    g_free(buffer);
    return program;
}

static cl_platform_id get_nvidia_platform(void)
{
    cl_platform_id *platforms = NULL;
    cl_uint num_platforms = 0;
    cl_platform_id nvidia_platform = NULL;

    cl_int errcode = clGetPlatformIDs(0, NULL, &num_platforms);
    if (errcode != CL_SUCCESS)
        return NULL;

    platforms = (cl_platform_id *) g_malloc0(num_platforms * sizeof(cl_platform_id));
    errcode = clGetPlatformIDs(num_platforms, platforms, NULL);

    gchar result[256];

    for (int i = 0; i < num_platforms; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 256, result, NULL);
        if (g_strstr_len(result, -1, "NVIDIA") != NULL) {
            nvidia_platform = platforms[i];
            break; 
        }
    }

    g_free(platforms);
    return nvidia_platform;
}

opencl_desc *ocl_new()
{
    opencl_desc *ocl = g_malloc0(sizeof(opencl_desc));

    cl_platform_id platform = get_nvidia_platform();
    if (platform == NULL)
        return NULL;

    int errcode = CL_SUCCESS;

    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl->num_devices));
    ocl->devices = g_malloc0(ocl->num_devices * sizeof(cl_device_id));
    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ocl->num_devices, ocl->devices, NULL));

    ocl->context = clCreateContext(NULL, ocl->num_devices, ocl->devices, NULL, NULL, &errcode);
    CHECK_ERROR(errcode);

    ocl->cmd_queues = g_malloc0(ocl->num_devices * sizeof(cl_command_queue));
    cl_command_queue_properties queue_properties = DO_PROFILE ? CL_QUEUE_PROFILING_ENABLE : 0;

    const size_t len = 256;
    char string_buffer[len];

    CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, len, string_buffer, NULL));
    printf("# Platform: %s\n", string_buffer);

    for (int i = 0; i < ocl->num_devices; i++) {
        CHECK_ERROR(clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, len, string_buffer, NULL));
        printf("# Device %i: %s\n", i, string_buffer);
        ocl->cmd_queues[i] = clCreateCommandQueue(ocl->context, ocl->devices[i], queue_properties, &errcode);
        CHECK_ERROR(errcode);
    }
    return ocl;
}

static void ocl_free(opencl_desc *ocl)
{
    for (int i = 0; i < ocl->num_devices; i++)
        clReleaseCommandQueue(ocl->cmd_queues[i]);

    CHECK_ERROR(clReleaseContext(ocl->context));

    g_free(ocl->devices);
    g_free(ocl->cmd_queues);
    g_free(ocl);
}

static void ocl_show_event_info(guint run, guint queue, guint event_type, guint num_events, cl_event *events)
{
    cl_ulong param;
    size_t size = sizeof(cl_ulong);

    g_print("# run device event [0=kernel,1=read] queued submitted start end\n");
    for (int i = 0; i < num_events; i++) {
        g_print("%d %d %d", run, queue, event_type);
        CHECK_ERROR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_QUEUED, size, &param, NULL));
        g_print(" %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_SUBMIT, size, &param, NULL));
        g_print(" %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, size, &param, NULL));
        g_print(" %ld", param);
        CHECK_ERROR(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, size, &param, NULL));
        g_print(" %ld\n", param);
    }
}

static Benchmark *setup_benchmark(opencl_desc *ocl)
{
    Benchmark *b = (Benchmark *) g_malloc0(sizeof(Benchmark));
    cl_int errcode = CL_SUCCESS;

    b->num_images = 30;
    b->width = 1024;
    b->height = 1024;
    b->image_size = b->width * b->height * sizeof(gfloat);
    b->single_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->multi_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->host_data = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->dev_data_in = (cl_mem *) g_malloc0(b->num_images * sizeof(cl_mem));
    b->dev_data_out = (cl_mem *) g_malloc0(b->num_images * sizeof(cl_mem));
    b->events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));
    b->read_events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));

    g_print("# Computing <nlm> for %i images of size %ix%i\n", b->num_images, b->width, b->height);

    for (guint i = 0; i < b->num_images; i++) {
        b->host_data[i] = (gfloat *) g_malloc0(b->image_size);
        b->single_result[i] = (gfloat *) g_malloc0(b->image_size);
        b->multi_result[i] = (gfloat *) g_malloc0(b->image_size);
        
        for (guint j = 0; j < b->width * b->height; j++)
            b->host_data[i][j] = (gfloat) g_random_double();

        b->dev_data_in[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b->image_size, b->host_data[i], &errcode);
        CHECK_ERROR(errcode);
        b->dev_data_out[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, b->image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
    }

    return b;
}

static void teardown_benchmark(Benchmark *b)
{
    for (int i = 0; i < b->num_images; i++) {
        g_free(b->host_data[i]);
        g_free(b->single_result[i]);
        g_free(b->multi_result[i]);
        CHECK_ERROR(clReleaseMemObject(b->dev_data_in[i]));
        CHECK_ERROR(clReleaseMemObject(b->dev_data_out[i]));
    }

    g_free(b->host_data);
    g_free(b->single_result);
    g_free(b->multi_result);
    g_free(b->dev_data_in);
    g_free(b->dev_data_out);
    g_free(b->events);
    g_free(b->read_events);
    g_free(b);
}

static void measure_single_gpu(Benchmark *benchmark, opencl_desc *ocl, cl_kernel *kernels)
{
    size_t global_work_size[2] = { benchmark->width, benchmark->height };
    GTimer *timer = g_timer_new();

    for (int i = 0; i < benchmark->num_images; i++) {
        CHECK_ERROR(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *) &benchmark->dev_data_in[i]))
        CHECK_ERROR(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *) &benchmark->dev_data_out[i]));
        
        CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[0], kernels[0],
                2, NULL, global_work_size, NULL,
                0, NULL, &benchmark->events[i]));

        CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[0], benchmark->dev_data_out[i], CL_FALSE, 0, 
                    benchmark->image_size, benchmark->single_result[i], 
                    1, &benchmark->events[i], &benchmark->read_events[i]));
    }

    clWaitForEvents(benchmark->num_images, benchmark->read_events);
    g_timer_stop(timer);
    g_print("# Single GPU: %fs\n", g_timer_elapsed(timer, NULL));
    g_timer_destroy(timer);

    if (DO_PROFILE) {
        ocl_show_event_info(0, 0, 0, benchmark->num_images, benchmark->events);
        ocl_show_event_info(0, 0, 1, benchmark->num_images, benchmark->read_events);
    }
}

static gdouble compare_single_multi(Benchmark *benchmark)
{
    gdouble sum = 0.0;
    for (guint i = 0; i < benchmark->num_images; i++)
        for (guint j = 0; j < benchmark->width * benchmark->height; j++)
            sum += ABS(benchmark->single_result[i][j] - benchmark->multi_result[i][j]);

    return sum;
}

static void measure_multi_gpu_single_thread(Benchmark *benchmark, opencl_desc *ocl, cl_kernel *kernels)
{
    const int batch_size = benchmark->num_images / ocl->num_devices;
    size_t global_work_size[2] = { benchmark->width, benchmark->height };
    GTimer *timer = g_timer_new();

    for (int i = 0; i < ocl->num_devices; i++) {
        for (int j = 0; j < batch_size; j++) {
            int idx = i*batch_size + j;
            CHECK_ERROR(clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *) &benchmark->dev_data_in[idx]))
            CHECK_ERROR(clSetKernelArg(kernels[i], 1, sizeof(cl_mem), (void *) &benchmark->dev_data_out[idx]));
            
            CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[i], kernels[i],
                        2, NULL, global_work_size, NULL,
                        0, NULL, &benchmark->events[idx]));

            CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[i], benchmark->dev_data_out[idx], CL_FALSE, 
                        0, benchmark->image_size, benchmark->multi_result[idx], 
                        1, &benchmark->events[idx], &benchmark->read_events[idx]));
        }
    }

    clWaitForEvents(benchmark->num_images, benchmark->read_events);
    g_timer_stop(timer);
    g_print("# %ix GPU (single thread): %fs (error=%f)\n", 
            ocl->num_devices, g_timer_elapsed(timer, NULL), compare_single_multi(benchmark));
    g_timer_destroy(timer);

    if (DO_PROFILE) {
        for (int i = 0; i < ocl->num_devices; i++) 
            ocl_show_event_info(1, i, 0, batch_size, benchmark->events + i*batch_size);

        for (int i = 0; i < ocl->num_devices; i++) 
            ocl_show_event_info(1, i, 1, batch_size, benchmark->read_events + i*batch_size);
    }
}

static gpointer thread_func(gpointer data)
{
    ThreadLocalBenchmark *tlb = (ThreadLocalBenchmark *) data;
    Benchmark *benchmark = tlb->benchmark;
    opencl_desc *ocl = tlb->ocl;
    cl_kernel *kernels = tlb->kernels;
    size_t global_work_size[2] = { benchmark->width, benchmark->height };
    const guint i = tlb->thread_id;

    for (int j = 0; j < tlb->batch_size; j++) {
        int idx = tlb->thread_id * tlb->batch_size + j;
        CHECK_ERROR(clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *) &benchmark->dev_data_in[idx]))
        CHECK_ERROR(clSetKernelArg(kernels[i], 1, sizeof(cl_mem), (void *) &benchmark->dev_data_out[idx]));
        
        CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[i], kernels[i],
                    2, NULL, global_work_size, NULL,
                    0, NULL, &benchmark->events[idx]));

        CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[i], benchmark->dev_data_out[idx], CL_FALSE, 
                    0, benchmark->image_size, benchmark->multi_result[idx], 
                    1, &benchmark->events[idx], &benchmark->read_events[idx]));
    }
}

static void measure_multi_gpu_multi_thread(Benchmark *benchmark, opencl_desc *ocl, cl_kernel *kernels)
{
    GThread *threads[ocl->num_devices];
    ThreadLocalBenchmark tlb[ocl->num_devices];
    const guint batch_size = benchmark->num_images / ocl->num_devices;

    g_thread_init(NULL);
    GTimer *timer = g_timer_new();

    for (guint i = 0; i < ocl->num_devices; i++) {
        tlb[i].benchmark = benchmark;
        tlb[i].ocl = ocl;
        tlb[i].kernels = kernels;
        tlb[i].batch_size = batch_size; 
        tlb[i].thread_id = i;

        threads[i] = g_thread_create(&thread_func, &tlb[i], TRUE, NULL);
    }

    /* Join first ... */
    for (guint i = 0; i < ocl->num_devices; i++)
        g_thread_join(threads[i]);

    /* ... wait on results last */
    clWaitForEvents(benchmark->num_images, benchmark->read_events);

    g_timer_stop(timer);
    g_print("# %ix GPU (multi thread): %fs (error=%f)\n", 
            ocl->num_devices, g_timer_elapsed(timer, NULL), compare_single_multi(benchmark));
    g_timer_destroy(timer);
}


int main(int argc, char const* argv[])
{
    g_print("## %s@%s\n", g_get_user_name(), g_get_host_name());

    cl_int errcode = CL_SUCCESS;
    opencl_desc *ocl = ocl_new();

    cl_program program = ocl_get_program(ocl, "nlm.cl", "");
    if (program == NULL) {
        g_warning("Could not open nlm.cl");
        ocl_free(ocl);
        return 1;
    }

    /* Create kernel for each device */
    cl_kernel *kernels = g_malloc0(ocl->num_devices * sizeof(cl_kernel));
    for (int i = 0; i < ocl->num_devices; i++) {
        kernels[i] = clCreateKernel(program, "nlm", &errcode);
        CHECK_ERROR(errcode);
    }

    Benchmark *benchmark = setup_benchmark(ocl);

    measure_single_gpu(benchmark, ocl, kernels);
    measure_multi_gpu_single_thread(benchmark, ocl, kernels);
    measure_multi_gpu_multi_thread(benchmark, ocl, kernels);

    teardown_benchmark(benchmark);

    g_free(kernels);
    ocl_free(ocl);
    return 0;
}


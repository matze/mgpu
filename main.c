
#include <CL/cl.h>
#include <glib-2.0/glib.h>
#include <stdio.h>

typedef struct {
    cl_context context;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_command_queue *cmd_queues;

    GList *kernel_table;
    GHashTable *kernels;         /**< maps from kernel string to cl_kernel */
} opencl_desc;

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

    gchar *buffer = (gchar *) g_malloc0(length);
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

opencl_desc *ocl_new(void)
{
    opencl_desc *ocl = g_malloc0(sizeof(opencl_desc));

    cl_platform_id platform;
    int errcode = CL_SUCCESS;
    CHECK_ERROR(clGetPlatformIDs(1, &platform, NULL));

    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl->num_devices));
    ocl->devices = g_malloc0(ocl->num_devices * sizeof(cl_device_id));
    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ocl->num_devices, ocl->devices, NULL));

    ocl->context = clCreateContext(NULL, ocl->num_devices, ocl->devices, NULL, NULL, &errcode);
    CHECK_ERROR(errcode);

    ocl->cmd_queues = g_malloc0(ocl->num_devices * sizeof(cl_command_queue));
    cl_command_queue_properties queue_properties = 0;

    const size_t len = 256;
    char device_name[len];
    for (int i = 0; i < ocl->num_devices; i++) {
        CHECK_ERROR(clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, len, device_name, NULL));
        printf("Device %i: %s\n", i, device_name);
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

int main(int argc, char const* argv[])
{
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

    /* Generate four data images */
    const int width = 1024;
    const int height = 1024;
    const size_t image_size = width * height * sizeof(float);
    const int num_images = 36;
    float **host_data = (float **) g_malloc0(num_images * sizeof(float *));
    cl_mem *dev_data_in = (cl_mem *) g_malloc0(num_images * sizeof(cl_mem));
    cl_mem *dev_data_out = (cl_mem *) g_malloc0(num_images * sizeof(cl_mem));

    g_print("Computing <nlm> for %i images of size %ix%i\n", num_images, width, height);

    for (int i = 0; i < num_images; i++) {
        host_data[i] = (float *) g_malloc0(image_size);
        dev_data_in[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image_size, host_data[i], &errcode);
        CHECK_ERROR(errcode);
        dev_data_out[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
    }

    cl_event events[num_images];
    cl_event read_events[num_images];
    size_t global_work_size[2] = { width, height };

    /* Measure single GPU case */
    GTimer *timer = g_timer_new();
    for (int i = 0; i < num_images; i++) {
        CHECK_ERROR(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), (void *) &dev_data_in[i]))
        CHECK_ERROR(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), (void *) &dev_data_out[i]));
        
        CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[0], kernels[0],
                2, NULL, global_work_size, NULL,
                0, NULL, &events[i]));

        CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[0], dev_data_out[i], CL_FALSE, 0, image_size, host_data[i], 1, &events[i], &read_events[i]));
    }

    clWaitForEvents(num_images, read_events);
    g_timer_stop(timer);
    g_print("Single GPU: %fs have passed\n", g_timer_elapsed(timer, NULL));

    /* Measure multiple GPU case */
    const int batch_size = num_images / ocl->num_devices;
    g_timer_start(timer);
    for (int i = 0; i < ocl->num_devices; i++) {
        for (int j = 0; j < batch_size; j++) {
            int idx = i*batch_size + j;
            CHECK_ERROR(clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *) &dev_data_in[idx]))
            CHECK_ERROR(clSetKernelArg(kernels[i], 1, sizeof(cl_mem), (void *) &dev_data_out[idx]));
            
            CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[i], kernels[i],
                        2, NULL, global_work_size, NULL,
                        0, NULL, &events[idx]));

            CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[i], dev_data_out[idx], CL_FALSE, 
                        0, image_size, host_data[idx], 
                        1, &events[idx], &read_events[idx]));
        }
    }

    clWaitForEvents(num_images, read_events);
    g_timer_stop(timer);
    g_print("%ix GPU: %fs have passed\n", ocl->num_devices, g_timer_elapsed(timer, NULL));
    g_timer_destroy(timer);

    for (int i = 0; i < num_images; i++) {
        g_free(host_data[i]);
        CHECK_ERROR(clReleaseMemObject(dev_data_in[i]));
        CHECK_ERROR(clReleaseMemObject(dev_data_out[i]));
    }
    g_free(kernels);
    g_free(host_data);
    g_free(dev_data_in);
    g_free(dev_data_out);

    ocl_free(ocl);
    return 0;
}


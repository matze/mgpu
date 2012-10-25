/* main.c as part of mgpu
 *
 * Copyright (C) 2011-2012 Matthias Vogelgesang <matthias.vogelgesang@gmail.com>
 *
 * mgpu is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * mgpu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Labyrinth; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

#include <CL/cl.h>
#include <glib.h>
#include <stdio.h>

#include "ocl.h"

typedef struct {
    gint num_images;
    guint width;
    guint height;
    gboolean do_profile;
} Settings;

typedef struct {
    Settings *settings;
    guint num_images;
    gsize image_size;
    gfloat **host_data;
    gfloat **single_result;
    gfloat **multi_result;
    cl_mem *dev_data_in;
    cl_mem *dev_data_out;
    cl_event *events;
    cl_event *read_events;
    cl_event *write_events;
    cl_kernel *kernels;
    opencl_desc *ocl;
} Benchmark;

typedef struct {
    Benchmark *benchmark;
    opencl_desc *ocl;
    guint batch_size;
    guint thread_id;
} ThreadLocalBenchmark;

typedef gboolean (*BenchmarkFunc) (Benchmark *);

static Benchmark *
setup_benchmark(opencl_desc *ocl, Settings *settings)
{
    Benchmark *b;
    cl_program program;
    cl_int errcode = CL_SUCCESS;

    program = ocl_get_program(ocl, "nlm.cl", "");

    if (program == NULL) {
        g_warning ("Could not open nlm.cl");
        ocl_free (ocl);
        return NULL;
    }

    b = (Benchmark *) g_malloc0(sizeof(Benchmark));
    b->ocl = ocl;
    b->settings = settings;

    /* Create kernel for each device */
    b->kernels = g_malloc0(ocl->num_devices * sizeof(cl_kernel));

    for (int i = 0; i < ocl->num_devices; i++) {
        b->kernels[i] = clCreateKernel(program, "nlm", &errcode);
        CHECK_ERROR(errcode);
    }

    b->num_images = b->settings->num_images < 0 ? ocl->num_devices * 16 : b->settings->num_images;
    b->image_size = b->settings->width * b->settings->height * sizeof(gfloat);
    b->single_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->multi_result = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->host_data = (gfloat **) g_malloc0(b->num_images * sizeof(gfloat *));
    b->events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));
    b->read_events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));
    b->write_events = (cl_event *) g_malloc0(b->num_images * sizeof(cl_event));

    g_print("# Computing <nlm> for %i images of size %ix%i\n", b->num_images, b->settings->width, b->settings->height);

    for (guint i = 0; i < b->num_images; i++) {
        b->host_data[i] = (gfloat *) g_malloc0(b->image_size);
        b->single_result[i] = (gfloat *) g_malloc0(b->image_size);
        b->multi_result[i] = (gfloat *) g_malloc0(b->image_size);

        for (guint j = 0; j < b->settings->width * b->settings->height; j++)
            b->host_data[i][j] = (gfloat) g_random_double();
    }

    b->dev_data_in = (cl_mem *) g_malloc0(ocl->num_devices * sizeof(cl_mem));
    b->dev_data_out = (cl_mem *) g_malloc0(ocl->num_devices * sizeof(cl_mem));

    for (guint i = 0; i < ocl->num_devices; i++) {
        b->dev_data_in[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, b->image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
        b->dev_data_out[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, b->image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
    }

    return b;
}

static void
teardown_benchmark (Benchmark *b)
{
    for (guint i = 0; i < b->num_images; i++) {
        g_free(b->host_data[i]);
        g_free(b->single_result[i]);
        g_free(b->multi_result[i]);
    }

    for (guint i = 0; i < b->ocl->num_devices; i++) {
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
    g_free(b->write_events);
    g_free(b);
}

static void
prepare_and_run_kernel (Benchmark *benchmark, guint device, guint index, gfloat *output)
{
    cl_command_queue cmd_queue = benchmark->ocl->cmd_queues[device];
    cl_kernel kernel = benchmark->kernels[device];
    cl_mem dev_data_in = benchmark->dev_data_in[device];
    cl_mem dev_data_out = benchmark->dev_data_out[device];
    cl_event *write_event_loc = &benchmark->write_events[index];
    cl_event *read_event_loc = &benchmark->read_events[index];
    cl_event *exec_event_loc = &benchmark->events[index];
    size_t global_work_size[2] = { benchmark->settings->width, benchmark->settings->height };

    CHECK_ERROR(clEnqueueWriteBuffer (cmd_queue, dev_data_in, CL_TRUE,
                0, benchmark->image_size, benchmark->host_data[index],
                0, NULL, write_event_loc));

    CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &dev_data_in));
    CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &dev_data_out));

    CHECK_ERROR(clEnqueueNDRangeKernel(cmd_queue, kernel,
                2, NULL, global_work_size, NULL,
                1, write_event_loc, exec_event_loc));

    CHECK_ERROR(clEnqueueReadBuffer(cmd_queue, dev_data_out, CL_FALSE,
                0, benchmark->image_size, output,
                1, exec_event_loc, read_event_loc));
}

static gdouble
compare_single_multi(Benchmark *benchmark)
{
    gdouble sum = 0.0;
    for (guint i = 0; i < benchmark->num_images; i++)
        for (guint j = 0; j < benchmark->settings->width * benchmark->settings->height; j++)
            sum += ABS(benchmark->single_result[i][j] - benchmark->multi_result[i][j]);

    return sum;
}

static gboolean
execute_single_gpu(Benchmark *benchmark)
{
    for (int i = 0; i < benchmark->num_images; i++)
        prepare_and_run_kernel (benchmark, 0, i, benchmark->single_result[i]);

    return FALSE;
}

static gboolean
execute_multi_gpu_single_thread(Benchmark *benchmark)
{
    const int batch_size = benchmark->num_images / benchmark->ocl->num_devices;

    for (int i = 0; i < benchmark->ocl->num_devices; i++) {
        for (int j = 0; j < batch_size; j++) {
            int idx = i*batch_size + j;
            prepare_and_run_kernel (benchmark, i, idx, benchmark->multi_result[idx]);
        }
    }

    return TRUE;
}

static gpointer
thread_func (gpointer data)
{
    ThreadLocalBenchmark *tlb = (ThreadLocalBenchmark *) data;
    Benchmark *benchmark = tlb->benchmark;
    const guint i = tlb->thread_id;

    for (int j = 0; j < tlb->batch_size; j++) {
        int idx = tlb->thread_id * tlb->batch_size + j;
        prepare_and_run_kernel (benchmark, i, idx, benchmark->multi_result[idx]);
    }

    return NULL;
}

static gboolean
execute_multi_gpu_multi_thread(Benchmark *benchmark)
{
    opencl_desc *ocl = benchmark->ocl;
    GThread *threads[ocl->num_devices];
    ThreadLocalBenchmark tlb[ocl->num_devices];
    const guint batch_size = benchmark->num_images / ocl->num_devices;

    for (guint i = 0; i < ocl->num_devices; i++) {
        tlb[i].benchmark = benchmark;
        tlb[i].ocl = ocl;
        tlb[i].batch_size = batch_size;
        tlb[i].thread_id = i;

        threads[i] = g_thread_create(&thread_func, &tlb[i], TRUE, NULL);
    }

    /* Join first ... */
    for (guint i = 0; i < ocl->num_devices; i++)
        g_thread_join(threads[i]);

    return TRUE;
}

static void
measure_benchmark (const gchar *prefix, BenchmarkFunc func, Benchmark *benchmark)
{
    gdouble time;
    gdouble error;
    gboolean check_error;
    GTimer *timer;
    
    timer = g_timer_new();
    check_error = func (benchmark);
    
    clWaitForEvents (benchmark->num_images, benchmark->read_events);
    g_timer_stop (timer);
    time = g_timer_elapsed (timer, NULL);
    g_timer_destroy(timer);

    error = check_error ? compare_single_multi (benchmark) : 0.0;
    g_print("# %s: total = %fs, time per image = %fs, error = %f\n",
            prefix, time, time / benchmark->num_images, error);
}

int main(int argc, char *argv[])
{
    static Settings settings = {
        .num_images = -1,
        .width = 1024,
        .height = 1024,
        .do_profile = FALSE
    };

    static GOptionEntry entries[] = {
        { "num-images", 'n', 0, G_OPTION_ARG_INT, &settings.num_images, "Number of images", "N" },
        { "width", 'w', 0, G_OPTION_ARG_INT, &settings.width, "Width of imags", "W" },
        { "height", 'h', 0, G_OPTION_ARG_INT, &settings.height, "Height of images", "H" },
        { "enable-profiling", 'n', 0, G_OPTION_ARG_NONE, &settings.do_profile, "Enable profiling", NULL },
        { NULL }
    };

    GOptionContext *context;
    opencl_desc *ocl;
    Benchmark *benchmark;
    GError *error = NULL;

    context = g_option_context_new (" - test multi GPU performance");
    g_option_context_add_main_entries (context, entries, NULL);

    if (!g_option_context_parse (context, &argc, &argv, &error)) {
        g_print ("Option parsing failed: %s\n", error->message);
        return 1;
    }

    g_print("## %s@%s\n", g_get_user_name(), g_get_host_name());

    g_thread_init (NULL);

    ocl = ocl_new (FALSE);
    benchmark = setup_benchmark (ocl, &settings);

    measure_benchmark ("Single GPU", execute_single_gpu, benchmark);
    measure_benchmark ("Single Threaded, Multi GPU", execute_multi_gpu_single_thread, benchmark);
    measure_benchmark ("Multi Threaded, Multi GPU", execute_multi_gpu_multi_thread, benchmark);

    teardown_benchmark(benchmark);

    ocl_free(ocl);
    return 0;
}

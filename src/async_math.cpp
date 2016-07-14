#include "async_math.h"

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <stdlib.h>

// Include the stuff for OpenCL
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

using namespace std;

const bool DEBUG = false;

// The log2(layers) in the partial sum array to run the sequential sum on
const int CUTOFF = 8;

// Store the kernel source code in an array of lines
const int SOURCE_LINES = 45;
const char* source[SOURCE_LINES] = {
  "#if CONFIG_USE_DOUBLE\n",
  "#ifdef cl_khr_fp64\n",
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
  "#define DOUBLE_SUPPORT_AVAILABLE\n",
  "#elif defined(cl_amd_fp64)\n",
  "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n",
  "#define DOUBLE_SUPPORT_AVAILABLE\n",
  "#else\n",
  "#error \"Double precision not supported.\"\n",
  "#endif\n",
  "#endif // CONFIG_USE_DOUBLE\n",
  "#if defined(DOUBLE_SUPPORT_AVAILABLE)\n",
  "typedef double real_t;\n",
  "#else\n",
  "typedef float real_t;\n",
  "#endif\n",
  "// kernel for performing multiplication of each a(i,j) and b(k,l) \n",
  "kernel void mat_mult(global real_t* a, global real_t* b, global real_t* mat_mult,\n",
  "                      const int a_rows, const int a_cols, const int b_cols) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  const size_t col = get_global_id(1);\n",
  "  const size_t layer = get_global_id(2);\n",
  "  const int idx = (layer * a_rows * b_cols) + (row * b_cols) + col;\n",
  "  mat_mult[idx] = a[(row * a_cols) + layer] * b[(layer * b_cols) + col];\n",
  "}\n"
  "// kernel for performing addition of layers\n",
  "kernel void part_sum(global real_t* mat_mult, const int a_rows, const int b_cols) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  const size_t col = get_global_id(1);\n",
  "  const size_t layer = get_global_id(2);\n",
  "  const size_t n = get_global_size(2);\n",
  "  const size_t add_layer = layer + n;\n",
  "  const int idx = (layer * a_rows * b_cols) + (row * b_cols) + col;\n",
  "  const int add_idx = (add_layer * a_rows * b_cols) + (row * b_cols) + col;\n",
  "  if (add_layer < a_rows)\n",
  "    mat_mult[idx] = mat_mult[idx] + mat_mult[add_idx];\n",
  "}\n"
  "// kernel for performing final addition of layers\n",
  "kernel void full_sum(global real_t* mat_mult, const int a_rows,\n",
  "                     const int b_cols, const int min_rows) {\n",
  "  const size_t row = get_global_id(0);\n",
  "  const size_t col = get_global_id(1);\n",
  "  real_t sum = 0.0;\n",
  "  for (int i = 0; i < min_rows; i++)\n",
  "    sum += mat_mult[(i * a_rows * b_cols) + (row * b_cols) + col];\n",
  "  mat_mult[(row * b_cols) + col] = sum;\n",
  "}\n"
};

// Open CL objects
char name[128];
cl_int err;
cl_platform_id platform;
cl_uint deviceCount;
cl_device_id device;
cl_context context;
cl_program program;
cl_kernel mat_mult_kernel;
cl_kernel part_sum_kernel;
cl_kernel full_sum_kernel;
cl_command_queue queue;
bool kernel_loaded = false;

// Loads the devices, etc for using Open CL
void AsyncMath::load_kernel() {
  if (!kernel_loaded) {
    // Get the platform
    clGetPlatformIDs(1, &platform, NULL);

    // Get the device count
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);

    // Get all of the devices
    cl_device_id devices[deviceCount];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

    // Choose the device with the most compute units
    int max_cus = 0;
    for (int i = 0; i < deviceCount; i++) {
      int this_cus = 0;
      long mem = 0;
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &this_cus, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(long), &mem, NULL);
      if (DEBUG) cout << "Device " << i << ": " << name << " (" << this_cus << " CUs) w " << mem << std::endl;

      if (this_cus > max_cus){
        device = devices[i];
        max_cus = this_cus;
      } // end if
    } // end for

    // Print out the device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    if (DEBUG) cout << "Using: " << name << std::endl;

    // Create the context for the device
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("cannot create device context");
    } // end if

    // Create the program from the source code
    program = clCreateProgramWithSource(context, SOURCE_LINES, source, NULL, &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("program could not be created from program source");
    } // end if

    // Build the program
    const char options[] = "-D CONFIG_USE_DOUBLE=false";
    err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("program could not be built");
    } // end if

    // Create the beta part kernel
    mat_mult_kernel = clCreateKernel(program, "mat_mult", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("mat mult kernel could not be created");
    } // end if

    // Create the beta part kernel
    part_sum_kernel = clCreateKernel(program, "part_sum", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("part sum kernel could not be created");
    } // end if

    // Create the beta part kernel
    full_sum_kernel = clCreateKernel(program, "full_sum", &err);
    if (err != CL_SUCCESS) {
      fprintf(stdout, "code: %d\n", err);
      throw std::runtime_error("full sum kernel could not be created");
    } // end if

    // Don't need to reload
    kernel_loaded = true;
  } // end if
} // end load_kernel

// Releases the devices, etc used by Open CL
void AsyncMath::release_kernel() {
  clReleaseKernel(mat_mult_kernel);
  clReleaseKernel(part_sum_kernel);
  clReleaseKernel(full_sum_kernel);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Make sure to reload next time
  kernel_loaded = false;
} // end release_kernel

void AsyncMath::MatrixMult (real_t * a, const int* a_dim, real_t * b, const int * b_dim, // input
                 real_t *& c, int *& c_dim // output
                 ) {
  // Get the dimensions
  const int a_rows = a_dim[0];
  const int a_cols = a_dim[1];
  const int b_rows = b_dim[0];
  const int b_cols = b_dim[1];

  // Initialize output memory
  real_t * c_tmp = new real_t[a_rows * b_cols];

  // Initialize computation memory
  real_t * mat_mult = new real_t[a_rows * b_cols * a_rows];

  // Load the OpenCL device stuff
  load_kernel();

  // Create the command queue to execute
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("command queue could not be created");

  // Set the input memory
  cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(real_t) * (a_rows * a_cols),  a, &err);
  cl_mem b_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(real_t) * (b_rows * b_cols),  b, &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("failed to allocate input buffer");

  // Set the input/output memory
  cl_mem mat_mult_io = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(real_t) * (a_rows * b_cols * a_rows), mat_mult, &err);
  if (err != CL_SUCCESS)
    throw std::runtime_error("failed to allocate i/o buffer");

  // Set scalar memory
  const cl_int a_rows_in = a_rows;
  const cl_int a_cols_in = a_cols;
  const cl_int b_cols_in = b_cols;
  const cl_int min_rows_in = (int)pow(2, CUTOFF);

  // Set the parameters
  // -- mat multiplication part
  clSetKernelArg(mat_mult_kernel, 0, sizeof(cl_mem), &a_in);
  clSetKernelArg(mat_mult_kernel, 1, sizeof(cl_mem), &b_in);
  clSetKernelArg(mat_mult_kernel, 2, sizeof(cl_mem), &mat_mult_io);
  clSetKernelArg(mat_mult_kernel, 3, sizeof(cl_int), &a_rows_in);
  clSetKernelArg(mat_mult_kernel, 4, sizeof(cl_int), &a_cols_in);
  clSetKernelArg(mat_mult_kernel, 5, sizeof(cl_int), &b_cols_in);
  // -- partial sums
  clSetKernelArg(part_sum_kernel, 0, sizeof(cl_mem), &mat_mult_io);
  clSetKernelArg(part_sum_kernel, 1, sizeof(cl_int), &a_rows_in);
  clSetKernelArg(part_sum_kernel, 2, sizeof(cl_int), &b_cols_in);
  // -- full sums
  clSetKernelArg(full_sum_kernel, 0, sizeof(cl_mem), &mat_mult_io);
  clSetKernelArg(full_sum_kernel, 1, sizeof(cl_int), &a_rows_in);
  clSetKernelArg(full_sum_kernel, 2, sizeof(cl_int), &b_cols_in);
  clSetKernelArg(full_sum_kernel, 3, sizeof(cl_int), &min_rows_in);

  // Initialize
  const int log_rows = (int)ceilf(log2f(a_rows));
  const int part_sums = log_rows - (CUTOFF - 1);
  const int mat_mult_dim = 3;
  const int part_sum_dim = 3;
  const int full_sum_dim = 2;
  const size_t mat_mult_dims[] = {a_rows, b_cols, a_rows};
  const size_t full_sum_dims[] = {a_rows, b_cols};

  // Multiplication: c(i,j,k) = a(i,k) * b(k,j)
  err = clEnqueueNDRangeKernel(queue, mat_mult_kernel, mat_mult_dim, NULL, mat_mult_dims, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    throw std::runtime_error("failed to queue mat mult");

  // Partial sums
  for (int s = 1; s < part_sums; s++) {
    const int layers = (int)pow(2, log_rows - s);
    const size_t part_sum_dims[] = {a_rows, b_cols, layers};
    err = clEnqueueNDRangeKernel(queue, part_sum_kernel, part_sum_dim, NULL, part_sum_dims, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
      throw std::runtime_error("failed to queue part sum");
  } // end for

  // full sums
  err = clEnqueueNDRangeKernel(queue, full_sum_kernel, full_sum_dim, NULL, full_sum_dims, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    throw std::runtime_error("failed to queue full sum");

  // Execute
  clFlush(queue);
  clFinish(queue);

  // Read out our results
  if (clEnqueueReadBuffer(queue, mat_mult_io, CL_TRUE, 0, sizeof(real_t) * a_rows * b_cols * a_rows, mat_mult, 0, NULL, NULL) != CL_SUCCESS)
    throw std::runtime_error("failed to read out mat_mult");

  // Extract results
  std::memcpy(c_tmp, mat_mult, sizeof(real_t) * a_rows * b_cols);

  // Clean up OpenCL resources
  clReleaseMemObject(a_in);
  clReleaseMemObject(b_in);
  clReleaseMemObject(mat_mult_io);

  // Clean up OpenCL resources (the rest)
  clReleaseCommandQueue(queue);
  release_kernel();

  // Set the return values
  int c_dim_temp[] = {a_rows, b_cols};
  c_dim = c_dim_temp;
  c = c_tmp;

  // Note: we don't delete [] c_temp since it is now pointed at by c
} // end MatrixMult

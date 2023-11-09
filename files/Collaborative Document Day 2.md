![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2

2023-11-06 GPU Programming.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

* [Google Colab](https://colab.research.google.com/)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Leon Oostrum

## 🧑‍🙋 Helpers

Giordano Lipari, Giulia Crocioni

----
## 🗓️ Agenda
|  Time | Topic                            |
| -----:|:-------------------------------- |
| 09:30 | Welcome and icebreaker           |
| 09:45 | Introduction to CUDA             |
| 10:30 | Coffee break                     |
| 10:45 | CUDA memories and their use      |
| 11:15 | Coffee break                     |
| 11:30 | CUDA memories and their use      |
| 12:00 | Lunch break                      |
| 13:00 | Data sharing and synchronization |
| 14:00 | Coffee break                     |
| 14:15 | Data sharing and synchronization |
| 15:00 | Coffee break                     |
| 15:15 | Concurrent access to the GPU     |
| 16:15 | Wrap-up                          |
| 16:30 | Drinks                           |
| 17:00 | END                              |

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises

The exercises are embedded in the Collaborative notes below

----
## 🧠 Collaborative Notes

* Summary of yesterday (all coding troubles seemed so far away)
    * Using libraries for scientific computing with a CPU
    * Using libraries for scientific computing with a GPU
    * Using Numba
        * just-in-time compilation
        * vectorization with CUDA as target
    * Limit: some performance with Python, without full exploitation of the (market) value of the graphics card.

### §5. [Your First GPU Kernel]()

* Overview of industry 

#### §5.1 Summing Two Vectors in Python

```python=
# 'hello world' equivalent in parallel computing
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
        # this is a mpa dataflow (see yesterday)
        


```
#### §5.2 Summing Two Vectors in CUDA

In the Jupyter notebook type CUDA code in a markdown cell. 
Jupyter does not run CUDA.

```cpp=
// this is C++/CUDA code and comments are prepended with // (like # in Python)

extern "C"  // topic not covered: CUDA linking extern C
__global__ void vector_add(const float * A, const float * B, float * C, const int size){
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
    # we do not use the variable size yet
}
```
* remarks on the meaning of \*, void, const, size

#### §5.3 Running Code on the GPU with CuPy

```python=
import cupy

# size of the vectors
size = 1024

# allocating and populating the vectors
# NB the type float32 is because the defaul float size in CUDA is 32 bit: 
# expect incorrect calculations when not taking into account this compatibility condition
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# embed CUDA vector_add in Python; r'''..''' indicates raw text
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

# this is the compilation action
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

# vector_add_gpu((a), (b), (c))
# c = arguments of CUDA function
# b = number of threads and how to organize them
# a = way the arguments in b are organized as groups
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))

# validation: check for correctness

import numpy as np

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
# test returns `array(True)`
```

#### §5.4 Understanding the CUDA Code

:::info
#### Discussion: Loose threads

We know enough now to pause for a moment and do a little exercise. Assume that in our vector_add kernel we replace the following line:

    int item = threadIdx.x;

With this other line of code:

    int item = 1;

What will the result of this change be?

1. Nothing changes
2. Only the first thread is working
3. Only C[1] is written
4. All elements of C are zero

##### Solution
3
:::

#### §5.5 Computing Hierarchy in CUDA

```python=
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
''', "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))

# this code throws an expected error
```

Discussion of CUDA programming model
* distinguish `__global__` (most frequent),`__device__`(infrequent), `__host__` (very seldom)
* grid, blocks, threads 
    * useful keywords

            threadIdx
            blockDim
            blockIdx
            gridDim

#### §5.6 Vectors of Arbitrary Size

To be commented
```python=
import math

grid_size = (int(math.ceil(size / 1024)), 1, 1)
block_size = (1024, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```
...

:::info
##### Challenge: Scaling up

In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).

    extern "C"
    __global__ void vector_add(const float * A, const float * B, float * C, const int size)
    {
       int item = ______________;
       C[item] = A[item] + B[item];
    }

###### Your answers

1. int item = blockDim.x * blockIdx.x + threadIdx.x;
2. int item = threadIdx.x + (blockIdx.x * blockDim.x);
3. blockIdx.x*blockDim.x + threadIdx.x
4. int item = blockIdx.x * blockDim.x + threadIdx.x;

###### Solution

```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   C[item] = A[item] + B[item];
}
```
:::

:::info
Break until 11:15
:::

We modify the vector_add kernel to include a check for the size of the vector, so that we only compute elements that are within the vector boundaries.

```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( item < size )
    {
        C[item] = A[item] + B[item];
    }
}
```

```python=
import cupy as cp
import math

size = 4_096  # first go with 4096, then with 4097
threads_per_block = 1024  # hardware constraint

a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( item < size) {
        C[item] = A[item] + B[item];
    }
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))

# validation test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

:::info
###### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python=
import numpy as np
import cupy
import math
from cupyx.profiler import benchmark
```

**CPU version**
```python=
def all_primes_to(upper : int, prime_list : list):
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)
```

**GPU version**
```python=
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
   for ( int number = 0; number < size; number++ )
   {
       int result = 1;
       for ( int factor = 2; factor <= number / 2; factor++ )
       {
           if ( number % factor == 0 )
           {
               result = 0;
               break;
           }
       }

       all_prime_numbers[number] = result;
   }
}
'''
```

**Allocate memory**
```python=
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)
```

**Setup the grid**
```python=
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
```

**Benchmark and test**

```python=
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)
execution_gpu = benchmark(all_primes_to_gpu, (grid_size, block_size, (upper_bound, all_primes_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the` check_prime_gpu_code` string, as we did in the examples so far.

Be aware that the CUDA code provided is a direct port of the Python code and, therefore, very slow. If you want to test it, user a lower value for  `upper_bound`.

###### Your answer

The table auto-formats while you fill it in, please bear with that!
The upside is that you do not have to fiddle around with formatting!

| Upper_limit | Execution time (ms) |
|:----------- | -------------------:|
| 100_000     |              39.954 |
| 100_000     |              38.711 |
| 100_000     |               38.69 |
| 100_000     |               33.93 |
| 100_000     |               39.92 |
| 100_000     |              38.693 |
| 100_000     |              38.696 |

###### Solution

```cpp=
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    // get rid of outer for loop
    // for ( int number = 0; number < size; number++ )
    int number = (blockId.x * blockDim.x) + threadIdx.x;
    //{
    int result = 1;
 
    if (number < size ) {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
            result = 0;
            break;
        }
        all_prime_numbers[number] = result;
    }     
   //}
}
```
:::

:::info
Lunch break until 13:00
:::

### §6 [Registers, Global, and Local Memory]()

#### §6.1 Registers

Baseline code:
```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size){
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```

Variant 1: missed
```cpp=
;
```

Variant 2: this another case when registers are well used
```cpp=
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size, const float factor){
    int item = threadIdx.x;
    float temp = A[item] + B[item];
    C[item] = temp * factor;
}
```

#### §6.2 Global Memory

This is the memory of the variables passed as arguments of the CUDA kernel.
On the GPU this is a read/write memory.

```cpp=
const float * A, const float * B, float * C, const int size, const float factor
```

#### §6.3 Constant Memory

* Read-only cache, once you moved that on the GPU.
* All threads can access it. It can be used to store input that gets broadcast to all threads.
* Just a few MB large.
* Fallen out of fashion in most recent architecture design, but still useful.

### §7 [Shared Memory and Synchronization]()

#### §7.1 Shared Memory

* To be declared inside the GPU
   ```cpp=
    int size = 128 // NB 128 is already largish
    __shared__ float small_array[size];
   ``` 
* Fast and small
* Close to the cores, less so than registers
* Sharing enables communication between threads, for example when checking for conditional instructions
* Sharing is within a block. Threads in different blocks are supposed to be impervious 

New notebook. 

Histogram function, like binning integers in bins of size 1.
Python baseline:
```python=
def histogram(input_array, output_array):
    # input_array is array of integers
    for item in input_array:
        output_array[item] = output_array[item] + 1

import numpy as np  # in a new notebook

# declare variables: 256 is because ...
input_array = np.random.randint(256, size=2048, dtype=np.int32)
output_array = np.zeros(256, dtype=np.int32)

#
histogram(input_array, output_array)
print(output_array)
```
CUDA version:
```cpp=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    output[input[item]] = output[input[item]] + 1;
}
```

:::info

##### Discussion: error in the histogram

```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    int temp = input[item]
    output[temp] = output[temp] + 1  // temp is not a thread index!
}
```
Each thread takes a unique *item*-indexed value and writes the output in a *temp*-index location, which could be the same for different thread. This can become a conflict.

###### Solution

Atomic operation: nobody interrupts anybody else; *locks* make each operation indivisible (atomic).
It will take longer but results are correct. No performance penalty in case of no conflict.

```cpp=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
```
:::

:::info
##### Challenge: use shared memory to speed up the histogram

Implement a new version of the CUDA `histogram` function that uses shared memory to reduce conflicts in global memory.

Modify the following code and follow the suggestions in the comments.

```c
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Declare temporary histogram in shared memory
    int temp_histogram[256];
    
    // Update the temporary histogram in shared memory
    atomicAdd();
    // Update the global histogram in global memory, using the temporary histogram
    atomicAdd();
}
```

##### Your answer
Type any character but an underscore when you are done
XXX

##### Solution

```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Declare temporary histogram in shared memory
    __shared__ int temp_histogram[256];
    
    // Update the temporary histogram in shared memory
    atomicAdd(&(temp_histogram[input[item]]), 1);
    // Update the global histogram in global memory, using the temporary histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```
* Resolving conflicts in shared memory is faster because of its physical location. The chance of conflict is also smaller because, in a general algorithm, smaller memory means fewer threads chancing to conflict.
* In first atomic add, the input decides which threads are used; the index is the integer `input[item]` 
* In the second atomic, all threads are used because the required operation spans across the full range of the histogram; this is faster because it is a bulk operation 

:::

#### §7.2 Thread Synchronization

```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
    
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();  // wait for all threads to stop 
    
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```
The above was also commented at the whiteboard.

One last missing link: start with a clean shared memory, as follows:

```cpp=
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
    
     // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();   
    
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();
    
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```

:::info
Break until 14:45
:::

### §9 [Concurrent access to the GPU]()

#### §9.1 Concurrently execute two kernels on the same GPU

```python=
import math
import numpy as np
import cupy
from cupyx.profiler import benchmark

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1

# input size
size = 2**25

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)


# CUDA code raw string from the latest version
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
histogram(input_cpu, output_cpu)  # CPU
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))  # GPU

if np.allclose(output_cpu, output_gpu):
    print("Correct results!")
else:
    print("Wrong results!")

# measure performance
%timeit -n 1 -r 1 histogram(input_cpu, output_cpu)
execution_gpu = benchmark(histogram_gpu, 
                          (grid_size, block_size,(input_gpu, output_gpu)),
                          n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")  # CPU 1mn 37s; GPU 2ms
```

:::info

###### Challenge: parallel reduction

Modify the parallel reduction CUDA kernel and make it work.

**Kernel**

To include in the Python code as `cuda_code = r'''...'''`
```cpp=
#define block_size_x 256
extern "C"
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively (with all threads in all thread blocks) iterate over input array
    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }

    //at this point we have reduced the number of values to be summed from n to
    //the total number of threads in all thread blocks combined

    //the goal is now to reduce the values within each thread block to a single
    //value per thread block for this we will need shared memory

    //declare shared memory array, how much shared memory do we need?
    //__shared__ float ...;

    //make every thread store its thread-local sum to the array in shared memory
    //... = sum;
    
    //now let's call syncthreads() to make sure all threads have finished
    //storing their local sums to shared memory
    __syncthreads();

    //now this interesting looking loop will do the following:
    //it iterates over the block_size_x with the following values for s:
    //if block_size_x is 256, 's' will be powers of 2 from 128, 64, 32, down to 1.
    //these decreasing offsets can be used to reduce the number
    //of values within the thread block in only a few steps.
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {

        //you are to write the code inside this loop such that
        //threads will add the sums of other threads that are 's' away
        //do this iteratively such that together the threads compute the
        //sum of all thread-local sums 

        //use shared memory to access the values of other threads
        //and store the new value in shared memory to be used in the next round
        //be careful that values that should be read are
        //not overwritten before they are read
        //make sure to call __syncthreads() when needed
    }

    //write back one value per thread block
    if (ti == 0) {
        //out_array[blockIdx.x] = ;  //store the per-thread block reduced value to global memory
    }
}
```

**Python host code**. 
This does not need to be modified for the exercise.

```python
# Allocate memory
size = numpy.int32(5e7)
input_cpu = numpy.random.randn(size).astype(numpy.float32) + 0.00000001
input_gpu = cupy.asarray(input_cpu)
out_gpu = cupy.zeros(2048, dtype=cupy.float32)

# Compile CUDA kernel
grid_size = (2048, 1, 1)
block_size = (256, 1, 1)
reduction_gpu = cupy.RawKernel(cuda_code, "reduce_kernel")

# Execute athe first partial reduction
reduction_gpu(grid_size, block_size, (out_gpu, input_gpu, size))
# Execute the second and final reduction
reduction_gpu((1, 1, 1), block_size, (out_gpu, out_gpu, 2048))

# Execute and time CPU code
sum_cpu = numpy.sum(input_cpu)

if numpy.absolute(sum_cpu - out_gpu[0]) < 1.0:
    print("Correct results!")
else:
    print("Wrong results!")
```

###### Your answer to: did you get correct results?

Edit the line below with your Y/N
NNNNNNNNNNNNN

###### Solution

```cpp=
#define block_size_x 256
extern "C"
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively (with all threads in all thread blocks) iterate over input array
    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }

    //at this point we have reduced the number of values to be summed from n to
    //the total number of threads in all thread blocks combined

    //the goal is now to reduce the values within each thread block to a single
    //value per thread block for this we will need shared memory

    //declare shared memory array, how much shared memory do we need?
    __shared__ float block sum[block_size_x];

    //make every thread store its thread-local sum to the array in shared memory
    block_sum[threadIdx.x] = sum;
    
    //now let's call syncthreads() to make sure all threads have finished
    //storing their local sums to shared memory
    __syncthreads();

    //now this interesting looking loop will do the following:
    //it iterates over the block_size_x with the following values for s:
    //if block_size_x is 256, 's' will be powers of 2 from 128, 64, 32, down to 1.
    //these decreasing offsets can be used to reduce the number
    //of values within the thread block in only a few steps.
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {

        if ( ti < s ) {
            sum = block_sum[ti] + block_sum[ti +s];
            block_sum[ti] = sum;
        }
        __syncthreads(); // never inside an if (unless all threads are inside the if block)
        
        //you are to write the code inside this loop such that
        //threads will add the sums of other threads that are 's' away
        //do this iteratively such that together the threads compute the
        //sum of all thread-local sums 

        //use shared memory to access the values of other threads
        //and store the new value in shared memory to be used in the next round
        //be careful that values that should be read are
        //not overwritten before they are read
        //make sure to call __syncthreads() when needed
    }

    //write back one value per thread block
    if (ti == 0) {
        out_array[blockIdx.x] = sum;  //store the per-thread block reduced value to global memory // sum is just the last value in the algoruthm
    }
}
```
* the number of operation is still $n$ but the number of *iterations* in the for loop is $\log n$ instead of $n$. So it is faster.
:::

### End of the course. Thanks for  your attending!
~~#### §9.2 Stream synchronization~~
~~#### §9.3 Measure execution time using streams and events~~

---

## :question: Questions

Add the questions you want to be answered in this section.


## 📚 Resources

* [CodiMD Markdown Guide](https://www.markdownguide.org/tools/codimd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
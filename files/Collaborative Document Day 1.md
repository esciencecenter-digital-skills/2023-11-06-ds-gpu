![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1

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

Giordano Lipari

---
## 🗓️ Agenda
|  Time | Topic                                               |
| -----:|:--------------------------------------------------- |
|  9:30 | Welcome and icebreaker                              |
|  9:45 | Introduction                                        |
| 10:00 | Convolve an image with a kernel on a GPU using CuPy |
| 10:30 | Coffee Break                                        |
| 10:45 | Running CPU/GPU agnostic code using CuPy            |
| 11:15 | Coffee break                                        |
| 11:30 | Image processing example with CuPy                  |
| 12:00 | Lunch break                                         |
| 13:00 | Image processing example with CuPy                  |
| 14:00 | Coffee break                                        |
| 14:15 | Run your Python code on a GPU using Numba           |
| 15:00 | Coffee break                                        |
| 15:15 | Run your Python code on a GPU using Numba           |
| 16:15 | Wrap-up                                             |
| 16:30 | END                                                 |

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building,   be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop, you can request a certificate of attendance by emailing to training@esciencecenter.nl.


---
## 🧠 Collaborative Notes

### §1. [Introduction]()

* Introduction of Netherlands eScience Center
* Presentation of GPU computing

### §2. [Using your GPU with CuPy]() = Convolve an image with a kernel on a GPU using CuPy

#### 2.1 Convolution in Python

```python=
import numpy as np
# create an image
deltas = np.zeros((2048, 2048))
deltas [8::16, 8::16] = 1  # start at (8,8) with stride 16

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(deltas[0:64, 0:64])  # a subset view out of convenience
ply.show()  # the pattern should be clear

# illustration of convolution
# illustration of map and stencil operations
```

##### 2.1.1 Convolution on the CPU Using SciPy

```python=
# generate a grid of independent variables to support the comvolution filter
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))  

# generate a Gaussian function (the convolution filter)
dst = np.sqrt(x**2 + y**2)
sigma = 1
mu = 0
gauss = np.exp(-(dst - mu)**2 / (2 * sigma**2))
plt.imshow(gauss)

# apply the convolution function in 2D from a library 
from scipy.signal import convolve2d as convolve2d_gpu

convolved_image_using_CPU = convolve2d_gpu(deltas, gauss)
plt.imgshow(convolved_image_using_CPU[:64, :64]) 

# timing the perfornabce with a Jupyter "magic"
%timeit -n 1 -r 1 convolve2d(deltas, gauss)     # 2.3 seconds = 2300 microseconds
```

:::info
   Break until 10:50 
:::

##### 2.1.2 Convolution on the GPU Using CuPy

```python=
# first requirement
# transfer data from CPU-based (host) memory to graphics-card (device) memory

import cupy as cp  # the GPU-oriented library

deltas_gpu = cp.asarray(deltas)  # this transfers the data
gauss_gpu = cp.asarray(gauss)

# pure scipy functions do not work on CPU, hence:
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
```
##### 2.1.3 Measuring performance

```python=
# REALITY CHECK 1: speed (aka profiling)

&timeit -n 7 -r 1 convolve2d_gpu(deltas_gpu, gauss_gpu)  # 0.16 milliseconds
2300 / 0.16   # ratio of %timeit results GPU / CPU: 10,000x is excessive

# cautionary note on asynchronous (=non-waiting) computing 
# advanced profiling function

from  cupyx.profiler import benchmark

execution_gpu = benchmark(convolve2d_gpu, (deltas_gpu, gauss_gpu),
                          n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")  # 218 microseconds
2300/220  # the performance gain: 100x is realistic 

```
:::info
###### Challenge: convolution on the GPU without CuPy**

Try to convolve the NumPy array deltas with the NumPy array gauss directly on the GPU, without using CuPy arrays. If this works, it should save us the time and effort of transferring deltas and gauss to the GPU.

###### Your answers
* Cannot construct a dtype from an array, due to memory access restrictions
* Cupy cannot build the proper dtype, assumignly because it cannot access the memory where the array is hosted.
* `TypeError: Cannot construct a dtype from an array`
* ""TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.""
:::

##### 2.1.4 Validation (correctness)

```python=
# REALITY CHECK 2: correctness

np.allclose(convolved_image_using_GPU, convolved_image_using_CPU)  # returns True as desired
```
:::info
###### Challenge: fairer runtime comparison CPU vs. GPU**

Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the `cp.asnumpy()` function.

###### Your answers (just the time with units please):
1. 0.0311 s, speedup 75.16 times
6. 0.0311 s
3. 0.0368 s
7. 0.0368 s
5. 0.0370 s
4. 0.0371 s
2. 0.0374 s
8. 0.0537 s (laptop, speedup ~80)


```python=
# solution
def transfer_compute_transferback():
    deltas_gpu = cp.asarray(deltas)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
    convolved_image_using_GPU_copied_to_host = cp.asnumpy(
        convolved_image_using_GPU)

execution_gpu = benchmark(transfer_compute_transferback, (), n_repeat=10)  # () no argument
gpu_avg_time = np.average(execution_gpu.gpu_times)  # 361 milliseconds
print(f"{gpu_avg_time:.6f} s")
2300/361  # 60x at minimal coding effort
```
:::

##### 2.1.5 NumPy routines on the GPU

```python=
# back to the host
convolve2d_cpu(deltas_gpu, gauss_gpu)  # expected fail because CPU/GPU memories are separate

deltas_1d = deltas.ravel()
gauss_1d = gauss.diagonal()
%timeit -n 1 -r 1 np.convolve(deltas_1d, gauss_1d)  # 212 ms

# back to the device
deltas_1d_gpu = cp.asarray(deltas_1d)
gauss_1d_gpu = cp.asarray(gauss_1d)

# profiling GPU-compatible
# note: numpy several functions run on cupy arrays works on GPU device
execution_gpu = benchmark(np.convolve, (deltas_1d_gpu, gauss_1d_gpu), n_repeat=10)

gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")  # 8 ms

212/8  # 25x speedup is realistic
```

#### 2.2 Application: image processing for radio astronomy

Instructor opens another Jupyter notebook

##### 2.2.1 Import the image

```python=
# import image using astronomy package

import os
from astropy.io import fits

teacher_dir = os.getenv('TEACHER_DIR')
fullpath = os.path.join(teacher_dir, 'JHS_data', 
                        'GMRT_image_of_Galactic_Center.fits')

# change of bit order in file (technicality)
with fits.open(fullpath) as hdul:
    data = hdul[0].data.byteswap().newbyteorder()
```
##### 2.2.2 Inspect the image

```python=
# plotting

# some imports are repeated because we are in another notebook 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm # typing onoing
%matplotlib inline

maxim = data.max()

# the following lines may differ from the instructor's, 
# but should work (notify if not, please)
fig = plt.figure(figsize=(50, 12.5))
ax = fig.add_subplot(1, 1, 1)
im_plot = ax.imshow(np.fliplr(data), cmap=plt.cm.gray_r, norm=LogNorm(vmin = maxim/10, vmax=maxim/100))  # plot of normalized data
plt.colorbar(im_plot, ax=ax)
plt.show()
```

##### 2.2.3 Step 1: The image background ($\kappa$-$\sigma$ clipping)

```python=
# inspect summary statistics of image

mean_ = data.mean()
median_ = np.median(data)
stddev_ = np.std(data)
max_ = np.amax(data)
print(f"{mean_:.2e}, {median_:.2e}, {stddev_:.2e}, {max_:.2e}")  
# orders of magnitude 10^-4 10^-5 10^-2 10^0 -- use for comparison

# flatten the 2d out of convenience
data_flat = data.ravel()

# kappa-sigma clipping for CPU
def kappa_sigma_clipper(data_flat):
    while True:
        # statistics (take note: numpy functions)
        med = np.median(data_flat)
        std = np.std(data_flat)
        # keep the data within the bounds
        clipped_lower = data_flat.compress(data_flat > med - 3 * std)
        clipped_both = clipped_lower.compress(clipped_lower < med + 3 * std)
        if len(clipped_both) == len(data_flat):
            break  # exit loop if the clipped dataset has no outliers
        data_flat = clipped_both  
    return data_flat

# note-taker's browser froze here

# check performance

data_clipped = kappa_sigma_clipper(data_flat)
timing_ks_clipping_cpu = %timeit -o kappa_sigma_clipper(data_flat)  # 530 ms
fastest_ks_clipping_cpu = timing_ks_clipping_cpu.best
print(f"Fastest CPU ks clipping time = \
       {1000 * fastest_ks_clipping_cpu:.3e} ms.")  # 520 ms

# check summary statistics of clipped dataset

clipped_mean_ = data_clipped.mean()
clipped_median_ = np.median(data_clipped)
clipped_stddev_ = np.std(data_clipped)
clipped_max_ = np.amax(data_clipped)
print(f"{clipped_mean_:.3e}, {clipped_median_:.3e}, 
      {clipped_stddev_:.3e}, {clipped_max_:.3e}")
# values not recorded
```

:::info
**Lunch break until 13:00**
:::

:::info
###### Challenge: κ,σ clipping on the GPU
Now that you understand how the κ-σ clipping algorithm works, perform it on the GPU using CuPy and compute the speedup.
Include the data transfer to and from the GPU in your calculation

###### Your answers (please type your speed-up):

The table auto-formats while you fill it in, please bear with that!

| Timing GPU in ms (optional) | Speed up (requested) |
| ---------------------------:|:--------------------:|
|                       70.22 |         7.37         |
|                       69.24 |          7.33           |
|                       69.97 |         7.37         |
|                       71.95 |         7.2          |
|                       65.75 |         7.11         | #"clipped_both = data_flat[(data_flat > med - 3*std) & (data_flat < med + 3*std)]"
|                       72.59 |         6.95         |
|                           t |          6.73           |

**Solution**
```python=
import cupy as cp

def ks_clipper_gpu(data_flat):
    # copy to GPU
    data_flat_gpu = cp.asarray(data_flat)
    # the function at the RHS contains only numpy functions
    # hence can work on the GPU when the input data are cupy objects
    data_gpu_clipped = kappa_sigma_clipper(data_flat_gpu)
    return cp.asnumpy(data_gpu_clipped)

# run

data_clipped_on_GPU = ks_clipper_gpu(data_flat)

# benchmark

from cupyx.profiler import benchmark

# second argument must be a tuple
timing_ks_clipping_gpu = benchmark(ks_clipper_gpu,
                                   (data_flat, ), n_repeat=10)

fastest_ks_clipping_gpu = np.amin(timing_ks_clipping_gpu.gpu_times)
print(f"{1000 * fastest_ks_clipping_gpu:.3e} ms")  # 75 ms

speedup_factor = fastest_ks_clipping_cpu/fastest_ks_clipping_gpu
print(f"The speedup factor for ks clipping is: 
      {speedup_factor:.3e}")  # 7x including data tranfer
```
:::

##### 2.2.4 Step 2: Segment the image (where are the sources?)

```python=
# baseline value
stddev_gpu_ = np.std(data_clipped_on_GPU)
print(f"standard deviation of background_noise = {stddev_gpu_:.4f} Jy/beam")

# use threshold to assign 0 where there is no source, else 1
threshold = 5 * stddev_gpu_
segmented_image = np.where(data > threshold, 1,  0)

# timing on CPU with %time magic 
# this is less intensive that k-s clipping
timing_segmentation_CPU = %timeit -o np.where(data > threshold, 1,  0)
fastest_segmentation_CPU = timing_segmentation_CPU.best 
print(f"Fastest CPU segmentation time = 
      {1000 * fastest_segmentation_CPU:.3e} ms.")  # 8ms

```

##### 2.2.5 Step 3: Label the segmented data (connected-component labelling)

Connected-component labelling (CCL) connects nearby pixels containing a source and assigns them an intensity level

```python=
# work on the CPU
from scipy.ndimage import label as label_cpu

labelled_image = np.empty(data.shape)  # create output as an empty array

number_of_sources_in_image = label_cpu(segmented_image, output = labelled_image)  
print(f"The number of sources level is {number_of_sources_in_image}.")  # 185

# timing on the CPU
timing_CCL_CPU = %timeit -o label_cpu(segmented_image, output = labelled_image)
fastest_CCL_CPU = timing_CCL_CPU.best
print(f"Fastest CPU CCL time = {1000 * fastest_CCL_CPU:.3e} ms.")  # 20 ms

# sanity-check: list of pixel intensity in the labelled image
np.unique(labelled_image)  # remove duplicates
```

##### 2.2.6 Step 4: Measure the source intensity

```python=
# handy functions from astronomy library (scipy for CPU)
from scipy.ndimage import center_of_mass as com_cpu
from scipy.ndimage import sum_labels as sl_cpu


all_positions = com_cpu(data, labelled_image,
                        range(1, number_of_sources_in_image+1))
all_integrated_fluxes = sl_cpu(data, labelled_image,
                               range(1, number_of_sources_in_image+1))

# next: not in the lesson materials

# look up the ten brightest sources
print(np.sort(all_integrated_fluxes)[-10:])  # range 38-360

# location of brightest source with timing oo
index_brightest = np.argmax(all_integrated_fluxes)  #  
all_positions[index_brightest]
```
Run the following in a single cell (note the **%%** magic command for the entire notebook cell, rather than **%** for a single line)

```python=
%%timeit -o
all_positions = com_cpu(data, labelled_image,
                        range(1, number_of_sources_in_image+1))
all_integrated_fluxes = sl_cpu(data, labelled_image,
                               range(1, number_of_sources_in_image+1))
# 917 ms
```
Continued:

```python=
# Fastest source measurement on CPU
timing_source_measurements_CPU = _
fastest_source_measurements_CPU = timing_source_measurements_CPU.best
print(f"{1000 * fastest_source_measurements_CPU:.2f} ms")
# 913 ms

# total processing time on CPU
total_time_CPU = fastest_ks_clipping_cpu + fastest_segmentation_cpu +
fastest_CCL_CPU + fastest_source_measurement CPU 
print(f"{1000 * total_time_CPU:.2f} ms")
# 1450 ms
```
:::info
###### Challenge: putting it all together

* **Numpy is mostly portable** Combine the first two steps of image processing for astronomy (determining background characteristics e.g. through $\kappa-\sigma$ clipping and segmentation) into a single function that works for **both** CPU and GPU. 
* **Performance** Next, write a function for connected component labelling and source measurements on the **GPU** and calculate the overall GPU-to-CPU speed-up factor for all four steps of image processing in astronomy. 
* **Verification** Finally, verify your output by comparing with the previous output fromthe CPU calculation (just above).
    * 913 ms fastest source measurement on CPU
    * 1450 ms total processing time on CPU


###### Your speed-ups
The table auto-formats while you fill it in, please bear with that!


| Speed up (requested) | Timing GPU ms (optional) |
| --------------------:|:------------------------ |
|                14.54 | 102.49                   |
|                12.42 | 120                      |
|                 11.8 | 122.8                    |
|                    s | 89.38                    |
|                12.37 | 119.398                  |

###### Solution

```python=
# work on CPU/GPU depending on input data for Numpy function

def first_two_steps_for_both_CPU_and_GPU(data):
    data_flat = data.ravel()
    # step 1 clipping
    data_clipped = kappa_sigma_clipper(data_flat)
    # step 2 segmenting
    stddev_ = np.std(data_clipped)
    threshold = 5 * stddev_
    segmented_image = np.where(data > threshold, 1,  0)
    return segmented_image

# work on the GPU

from cupyx.scipy.ndimage import label as label_gpu
from cupyx.scipy.ndimage import center_of_mass as com_gpu
from cupyx.scipy.ndimage import sum_labels as sl_gpu

def ccl_and_source_measurements_on_GPU(data_GPU, segmented_image_GPU):
    # step 3
    labelled_image_GPU = cp.empty(data_GPU.shape)
    number_of_sources_in_image = label_gpu(segmented_image_GPU, 
                                           output= labelled_image_GPU)
    # step 4
    all_positions = com_gpu(data_GPU, labelled_image_GPU, 
                            cp.arange(1, number_of_sources_in_image+1))
    all_fluxes = sl_gpu(data_GPU, labelled_image_GPU, 
                            cp.arange(1, number_of_sources_in_image+1))
    # convert lists into arrays and return arrays to CPU memory
    return cp.asnumpy(cp.asarray(all_positions)), cp.asnumpy(cp.asarray(all_fluxes))

# another function that combines steps 1 through 4

def run_all_GPU(data):
    data_GPU = cp.asarray(data)
    segmented_image_GPU = first_two_steps(data_GPU)
    return ccl_and_source_measurements(data_GPU, segmented_image_GPU)

# timing/profiling on the GPU

timing_total_GPU = benchmark(run_all_GPU, (data, ), n_repeat=10)
print(f"Total processing time on GPU: {1000 * np.admin(timing_total_GPU.gpu_times:/2f)} ms")  # 95 ms
1450 / 95 # seped-up
```
:::

:::info
Back at 14:25
:::

### §3 [Accelerate your Python code with Numba]()

New notebook!

#### 3.1 Numba to execute Python on GPU

```python=
# a function that does something at a compute cost

def find_all_primes_cpu(upper):
    # gives all prime numbers between 1 and upper
    # operates in a naive Python way
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        # shortcut: no number is divisible from a number larger than its half
        for i in range(2, (num // 2) + 1):  
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers

find_all_primes_cpu(upper)

# 10000 is a reasonable number for a Jupyter notebook
%timeit -n 10 -r 1 find_all_primes_cpu(10_000)  # 164 ms on CPU

# using Numba on CPU

import numba as nb

# decorator notation 
@nb.jit(nopython=True)
def find_all_primes_cpu(upper):
    # gives all prime numbers between 1 and upper
    # operates in a naive Python way
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        # shortcut: no number is divisible from a number larger than its half
        for i in range(2, (num // 2) + 1):  
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers

# non-decorator notation of the same 
# !!! TO BE DOUBLE CHECKED !!!
# find_all_primes_cpu = nb.jit(nopython=True)(find_all_primes_cpu)

# timimg with Numba on CPU


%timeit -n 10 -r 1 find_all_primes_cpu(10_000)  
# 7.3 ms after first-time compilation and having created a fast executable

# using Numba on GPU

from numba import cuda  # numba is presently supporting Nvidia cards

# do not use loops only checks on a single occurrence
@nb.cuda.jit(nopython=True)
def check_prime_gpu_kernel(num, result):
    # returns the input if the num is prime, else 0 
    result[0] =  num
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           result[0] = 0
           break

import numpy as np  # because it is a new notebook
result = np.zeros(1, np.int32)  # initialize output (a scalar)

# arguments between [] will be clear tomorrow
check_prime_gpu_kernel[1, 1](11, result)  # input is a prime
# one-time performance warning: under-utilization
check_prime_gpu_kernel[1, 1](12, result)  # input is not a prime
```

:::info
###### Challenge: compute prime numbers

Write a new function `find_all_primes_cpu_and_gpu` that uses `check_prime_gpu_kernel` instead of the inner loop of `find_all_primes_cpu`. 
How long does this new function take to find all primes up to 10,000?

###### Your answer
The table auto-formats while you fill it in, please bear with that!
Edit the pre-filled values that do not apply to you

| Speed up (optional) | Timing GPU ms (REQUESTED) | Threads | Upper limit |     |
|:------------------- | -------------------------:|:------- |:----------- | --- |
| 0.0011              |                      6830 | 1000    | 10_000      |     |
| 0.035               |                    4580 | 1       | 10_000      |     |
|                     |                      4490 | 1       | 10_000      |     |
| 0.0016              |                      4420 | 1       | 10_000      |     |
| s                   |                      4180 | 1       | 10_000      |     |
| s                   |                      3980 | 1       | 10_000      |     |
| 0.66                |                       596 | 1       | 10_000      |     |
| s                   |                       590 | 1       | 10_000      |     |
| s                   |                       583 | 1       | 10_000      |     |

###### Solution

```python=
def find_all_primes_cpu_and_gpu(upper):
    # the input is the upper limit of a range
    all_prime_numbers = []
    for num in range(0, upper):
        result = np.zeros(1, np.int32)
        check_prime_gpu_kernel[1,1](num, result)
        if result[0] != 0:
            all_prime_numbers.append(num)
    return all_prime_numbers

# timing on the GPU (numba is synchronous (waiting), unlike cupy )
%timeit -n 10 -r 1 find_all_primes_cpu_and_gpu(10_000)  # 4550 ms (!)
```
:::

```python=
def find_all_primes_gpu(numbers, result):
    # the input is an array
    num = numbers[cuda.threadIdx.x]  # 
    result[cuda.threadIdx.x] = num
    for i in range(2, (num // 2) + 1):
        if (num % i) == 0:
            result[cuda.threadIdx.x] = 0
            break

# run function on a 1024-item array with a GPU with 1024 threads

numbers = np.arange(0, 1024, dtype=np.int32)
result = np.zeros(1024, np.int32)
find_all_primes_gpu[1,1024](numbers, result)
printresult[:10])

# compare timing

%timeit -n 10 -r 1 find_all_primes_cpu_gpu(1024) # 418 ms ; input is upper limit
%timeit -n 10 -r 1 find_all_primes_gpu[1,1024](numbers, result) # 0.956 ms
```

### §3. [A Better Look at the GPU]()

```python=
# vectorize on GPU with Numba

@nb.vectorize(['int32(int32)'], target='cuda')  
def check_prime_gpu(num): 
    # warning! the block of a vectorized function is still a for-loop
    # But the input can also be an array and it will work elementwise 
    # Note: n!
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           return 0
    return num

# timing (warning repeated: the input of the vectorized function can be an array!)

%timeit -n 10 -r 1 check_prime_gpu(np.arange(1024, dtype=np.int32))  
# 0.817 ms with performance warning

%timeit -n 10 -r 1 check_prime_gpu(np.arange(10_000, dtype=np.int32))  
# 2.3 ms with performance warning

%timeit -n 10 -r 1 check_prime_gpu(np.arange(100_000, dtype=np.int32))  
# 83 ms without performance warning
```
:::info
Break until 15:45
:::

* Notes on levels of memory and their accessibility for you
---

## :question: Questions

Add the questions you want to be answered in this section.
* How to handle data too large for GPU Memory?
    * GPUs memories are larger than they used to be, but still smaller than the amount of RAM in most systems. Unfortunately there is no magic way available to the general public to address this issue, so it is up to you as a programmer to e.g. split the data in chunks and process them on the GPU. If the processing time is the bottleneck, you will already have a way to speed up your code. If not, it is also possible to copy data to or from the GPU while at the same time use the GPU to compute, so that you can hide the transfer times.
    * In short, it is up to you.
     
---
## 📚 Resources

* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
# Project Proposal (Group U) 
## 1. Project Goal 
This project focuses on developing an efficient object detection system for images, utilizing 
vector matching techniques instead of traditional machine learning models. The core idea 
is to identify objects by comparing feature vectors extracted from input images against a 
pre-built database of reference vectors. We aim to solve the problem of computationally 
expensive object detection in resource-constrained environments, where deep neural 
networks might be overkill or impractical due to their high training and inference demands. 
we target building a scalable system that can process large volumes of images quickly, 
making it suitable for applications in edge computing or mobile devices where GPU 
resources are available but limited. 
## 2. Use of GPU Computing 
GPUs are essential to this project because object detection via vector matching involves 
intensive computations that benefit immensely from parallel processing. Specifically, the 
matching phase requires calculating similarity scores e.g. such as cosine similarity or 
Euclidean distance and that too between thousands of feature vectors from a query image 
and a database containing potentially millions of reference vectors. This creates a highly 
parallelizable workload, where each comparison can be executed independently across 
GPU threads. For instance, in a single image with 1,000 keypoints and a database of 10,000 
vectors, we could perform up to 10 million operations, which would be bottlenecked on a 
CPU due to sequential processing. GPUs excel here through massive parallelism, allowing 
us to assign thread blocks to subsets of vectors, utilizing shared memory for faster data 
access and reducing latency.  
## 3. Expected Outcome 
We aim to deliver a working prototype that accurately detects objects in test images. The 
system will include a CPU baseline for comparison, highlighting metrics such as execution 
time, throughput, and resource utilization. Expected results include a detailed report 
quantifying speedup factors, accuracy evaluations using metrics like intersection over 
union (IoU), and visualizations of detection outputs. Data for validation will come from 
public sources like the MS COCO dataset. Ultimately, this project will prove the viability of 
vector-based detection on GPUs. 

#### Team Members: Muhammad Zahid, Sadikshya Satyal, Ayodeji Ibrahim, Ashir Kulshreshtha
# ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context

The source code for *ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context*

http://47.93.42.104/215 (Cause of death: CVD)   
http://47.93.42.104/318 (Cause of death: GI disease)   
http://47.93.42.104/616 (Cause of death: Other)   
http://47.93.42.104/265 (Cause of death: GI disease)    
http://47.93.42.104/812 (Cause of death: Cachexia)   
http://47.93.42.104/455 (Cause of death: CVD)       
http://47.93.42.104/998 (Alive)       
http://47.93.42.104/544 (Alive)    

Thanks for your interest in our work. Welcome to test the prototype of our visualization tool. The clinical hidden status is built by our latest representation learning model ConCare. The internationalised multi-language support will be available soon.

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Data preparation
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run decompensation prediction task on MIMIC-III bechmark dataset, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the **in-hospital mortality** dataset, please save the files in ```in-hospital-mortality``` directory to ```data/``` directory.


To make it easier for you to use our code as well as the data, we have uploaded the trained model as well.  We also upload the demographic (static baseline) information files to the Google Drive. 

You can download it from this link directly. https://drive.google.com/file/d/1TXn4UdtQCzfd7TdDJAo_6_IcnO2LUa1a/view?usp=sharing

The test set can be obtained via https://drive.google.com/file/d/1KHRPLznKMFi4s1hCxDxAjGWBPPgHXxj7/view?usp=sharing

## Run ConCare

All the hyper-parameters and steps are included in the `.ipynb` file, you can run it directly.

You can also load the trained model, which is saved in the `concare0` file.

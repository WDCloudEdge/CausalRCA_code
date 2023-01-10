# CausalRCA_code

## Description

This repository includes codes and data for CausalRCA. 

## Data 

We deploy the sock-shop with Kubernetes on several VMs in the cloud and inject anomalies to simulate performance issues of a running microservice application.

We collect data with *data_collection_all_services.ipynb* and put all data in the folder *data_collected*. 

We collect both service-level and resource-level data. At the service level, we collect the latency of each service. At the resource level, we collect container resource-related metrics, including CPU usage, memory usage, disk read and write, and network receive and transmit bytes.


## Code 

We provide codes for benchmark methods and CausalRCA, including the three experiments in our paper for latency, single-serve, and full-serve tests.

For benchmark test, run *bench_test-latency.ipynb*, *bench_test-single_service.ipynb*, or *bench_test-all_service.ipynb*.

For CausalRCA, run *train_latency.py*, *train_single_service.py*, or *train_all_services.py*. 

# ICSE-2027
This is data and code for ProbeFuse submited to ICSE-2027



## Instructions
This is the experimental process code and data of the proposed method Coding-Fuse and FMF method, which can be found in the code directory.

If you need to repeat the performance values ​​of the experimental results, we recommend running the test code directly, in each setcode parameter --test part. But before that, you need to (1) first copy the dataset folder to the directory at the same level as code in each fusion folder. (2) Secondly, import the code pre-trained models in the code folder, and we have prepared the file name for you, prepared code PTMs can be found at https://zenodo.org/records/15543665 (3) Finally, you need to import our trained test model (can be found at https://zenodo.org/records/15543870, https://zenodo.org/records/15544997 and https://zenodo.org/records/15546108) in saved_models in the code folder. You can get the same performance values ​​as us.

If you need to perform the entire experiment, that is, the entire process from training to testing, we recommend that you directly execute all the commands of setcode, but before that you also need to perform the first two steps above.

## Details
### Dataset Partition
code clone detection: Train-901028, Valid-415416, Test-415416

code smell detection: Train-1089, Valid-350, Test-350

technical debt detection: Train-24034, Valid-7674, Test-6652
### Finetune 

For code clone detection, use 10% Train and 10% Valid for fine-tuning, for code smell detection and technical debt detection, use full Train and Valid for fine-tuning.

### Hyperparameters
The training and testing commands and hyperparameters can be found in the setcode files in each folder. Training hyperparameters such as learning rate, batch size, and random seed are aligned with FMF and OSM in the same task and scenario to ensure fair and optimal comparisons.


If you have any questions, you can leave a message.


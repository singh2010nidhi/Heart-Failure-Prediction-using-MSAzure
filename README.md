# Heart Failure Prediction using Microsoft Azure

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs. People with cardiovascular disease or who are at high cardiovascular risk need early detection and management wherein a machine learning model can be of great help. This project involves training Machine Learning Model to predict mortality by Heart Failure using Microsoft Azure and deployment of the model as a web service. We also figure the main factors that cause mortality.

## Project Architecture

The following diagram shows the overall architecture and workflow of the project.

![](images/Project_Architecture.png)

## Project Details
* [Project Set Up and Installation](#project-set-up-and-installation)
* [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
* [Automated ML](#automated-ml)
  * [Results](#results)
* [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Results](#results)
* [Model Deployment](#model-deployment)
* [Screen Recording](#screen-recording)
* [Standout Suggestions](standout-suggestions)
* [Improvements and Future Work](#improvements-and-future-work)
* [References](#references)

## Project Set Up and Installation

Firstly, we need an Azure subscription to access the Azure workspace. For this project, the Azure subscription provided by Udacity was used.

### Create a Workspace

The workspace is the top-level resource that provodes a centralized place to work with all the artifacts we create when we use Azure Machine Learning. The workspace keeps a history of all training runs, including logs, metrics, output, and a snapshot of our scripts. 

The workspace can be created with the help of [Create and manage Azure Machine Learning workspaces](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python) document.

### Set up Compute Instance

A compute instance is a managed cloud-based workstation which is used as a fully configured and managed development environment. 

A compute instance with name `notebook139012` and virtual machine size of `STANDARD_DS3_V2` was created.

The screenshot below shows the registered compute instances.

![](images/Compute_Instance.png)

### Set up Compute Cluster

Compute cluster is a managed-compute infrastructure that allows us to easily create a single or multi-node compute. Compute clusters scales up automatically when a job is submitted and can run jobs securely in a virtual network environment.

A compute cluster `new-compute` with virtual machine size of `STANDARD_D2_V2` and `max_nodes =4` was created.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Improvements and Future Work


## References

# CBMS 2020 Paper submission
Code repository for "PRINCESS: Prediction of Individual Breast Cancer Evolution to Surgical Size"  by Cristian Axenie and Daria Kurz submitted at IEEE CBMS 2020, Mayo Clinic, Rochester, USA, July 28 - 30, 2020.

## Abstract
Modelling surgical size is not inherently meant to replicate the tumor's exact form and proportions, but instead to elucidate the degree of the tissue volume that may be surgically removed in terms of improving patient survival and minimize the risk that a second or third operation will be needed to eliminate all malignant cells entirely. Given the broad range of models of tumor growth, there is no specific rule of thumb about how to select the most suitable model for a particular breast cancer type and whether that would influence its subsequent application in surgery planning. Typically, these models require tumor biology-dependent parametrization, which hardly generalizes to cope with ttumor heterogeneity. In addition, the datasets are limited in size owing to the restricted or expensive methods of measurement. We address the shortcomings that incomplete biological specifications, the variety of tumor types and the limited size of the data bring to existing mechanistic tumor growth models and introduce a Machine Learning model for the PRediction of INdividual breast Cancer Evolution to Surgical Size (PRINCESS). This is a data-driven model based on neural networks capable of unsupervised learning of cancer growth curves. PRINCESS learns the temporal evolution of the tumor along with the underlying distribution of the measurement space. We demonstrate the superior accuracy of PRINCESS, against four typically used tumor growth models, in extracting tumor growth curves from a set of nine clinical breast cancer datasets. Our experiments show that, without any modification, PRINCESS can learn the underlying growth curves being versatile between breast cancer types.

### PRINCESS: Prediction of Individual Breast Cancer Evolution to Surgical Size


PRINCESS Codebase:

* datasets - the experimental datasets (csv files) and their source, each in separate directories
* models   - codebase to run and reproduce the experiments


Directory structure:

*models/PRINCESS/.*

* create_init_network.m       - init PRINCESS network (SOM + HL)
* error_std.m                 - error std calculation function
* princess_core.m             - main script to run PRINCESS
* model_rmse.m                - RMSE calculation function 
* model_sse.m                 - SSE calculation function
* parametrize_learning_law.m  - function to parametrize PRINCESS learning
* present_tuning_curves.m     - function to visualize PRINCESS SOM tuning curves
* randnum_gen.m               - weight initialization function
* tumor_growth_model_fit.m    - function implementing ODE models
* tumor_growth_models_eval.m  - main evaluation on PRINCESS runtime
* visualize_results.m         - visualize PRINCESS output and internals
* visualize_runtime.m         - visualize PRINCESS runtime



Usage: 

* **models/PRINCESS/princess_core.m** - main function that runs PRINCESS and generates the runtime output file (mat file)
* **models/PRINCESS/tumor_growth_models_eval.m** - evaluation and plotting function reading the PRINCESS runtime output file



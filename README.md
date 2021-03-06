# PRINCESS: Prediction of Individual Breast Cancer Evolution to Surgical Size

*Code repository for "PRINCESS: Prediction of Individual Breast Cancer Evolution to Surgical Size"  by Cristian Axenie and Daria Kurz submitted at IEEE CBMS 2020, Mayo Clinic, Rochester, USA, July 28 - 30, 2020.*

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



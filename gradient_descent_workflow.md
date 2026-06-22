# MM_QP Execution Flow

This document outlines the step-by-step execution chain when calling the `MM_QP` model from the wrapper. 

### The Entry Point: MM_QP
* Acts as a shortcut and wrapper for the main execution pipeline.
* Triggers the `run_MM_QP` function.

### Data Preparation: run_MM_QP
* Copies the parameter object to prevent overwriting the original input data.
* Calls **input_AMN** to format the input matrix **X** and append dummy columns for constraints.
* Calls **QP_layers** to initiate the actual optimization process.

### Orchestration: QP_layers
* Manages the model initialization and sets up the gradient descent loop.
* Calls **get_V0** to calculate the initial flux vector.
* Calls **Gradient_Descent** to run the primary optimization loop.
* Calls **output_AMN** to format the final output vector.

### The Engine: Gradient_Descent 
* Executes the core simulation by looping for a defined number of epochs.
* Calls **Loss_all** during each iteration to calculate the loss and gradients.
* Triggers specific loss functions (**Loss_Vout_constraint**, **Loss_SV**, **Loss_Pin**, and **Loss_Vpos**) via `Loss_all`.
* Updates the flux vector **V** using the newly calculated gradient **dL**.
* Calls **custom_ReLU** if the `hardConst` parameter is set to 2.

### The Return Path
* **QP_layers** returns the final outputs (the full state vector) and loss statistics back up the chain.
* **run_MM_QP** processes these outputs by slicing them to extract just the fluxes (**Vf**).
* Calculates the **R2 scores** to evaluate the fit.
* Writes the loss and target data to output CSV files.
* Returns the final flux matrix **Vf** and the `Stats` object to the user.

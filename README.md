# AAAI_code
This repository contains the code associated with the paper submitted to the Conference and Workshop on Association for the Advancement of Artificial Intelligence (AAAI).
## Data
The raw experimental data is stored in the `M` and `ORLIB` folders. To prepare the input file required for the experiment, which adheres to the CL restrictions, the raw data can be processed using the provided interface in the `data_deal_FL.py` file. This script ensures that the raw data is converted into the appropriate format for subsequent experimental use. Similarly, for clustered datasets, "data_deal_clustering.py" is used for processing.
## License
The code is released under the GNU General Public License, version 3, or any later version as published by the Free Software Foundation.
## Usage
Download the code.
### Steps to Test the Dataset

1. **Generate Input Data:**  
   Run “data_deal_FL.py” to process the dataset and generate 100 input files for each percentage of CL constraints.

2. **Run the Proposed Algorithm:**  
   Execute `Facility_CL.py` to obtain the results of our proposed algorithm.

3. **Run the Baseline Algorithm:**  
   Execute `compare_Greedy.py` and  `compare_Greedy_matching.py` to obtain the results of the baseline algorithms for comparison.


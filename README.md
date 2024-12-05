# WWW_code
This repository contains the code associated with the paper submitted to the ACM TheWebConf 2025 conference (WWW). 
## Data
The raw experimental data is stored in the `original_data` folder. To prepare the input file required for the experiment, which adheres to the CL restrictions, the raw data can be processed using the provided interface in the `src/data_deal.py` file. This script ensures that the raw data is converted into the appropriate format for subsequent experimental use.
## License
The code is released under the GNU General Public License, version 3, or any later version as published by the Free Software Foundation.
## Usage
Download the code.
### Steps to Test the Dataset

1. **Generate Input Data:**  
   Run `src/data_deal.py` to process the dataset and generate input files with CL constraints.

2. **Run the Proposed Algorithm:**  
   Execute `src/facility_location.py` to obtain the results of our proposed algorithm.

3. **Run the Baseline Algorithm:**  
   Execute `src/compare.py` to obtain the results of the baseline algorithms for comparison.



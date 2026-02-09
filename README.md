# A-Centrality-independent-Framework-for-Higher-Order-Cumulants-in-Heavy-Ion-Collisions
Paper: A Centrality-independent Framework for Revealing Genuine Higher-Order Cumulants in Heavy-Ion Collisions

#########################################################################################
#####################################  Author: Zhaohui Wang    ##########################
## Paper: 	A Centrality-independent Framework for Revealing Genuine Higher-Order Cumulants in Heavy-Ion Collisions
#########################################################################################

## Running Instructions
You need to prepare these parameters in `Run.py`:
<!-- ScanMax = False
RunResults = True
ScanOnlyC3 = False
bin_ref3[:-1] = bin_ref3[:-1] + 1
NCENT = len(bin_ref3)-2
START_FIT = input("Please input the start fit bin: ")
MAX = input("Please input the max fit bin: ")
LAST_C3 = input("Please input the last C3 value: ")
ENERGY = float(input("Please input the energy: "))
average_weight = float(input("Please input the average weight: ")) -->

### How to prepare parameters:
- `START_FIT`: The bin number to start the fitting.
- `MAX`: The maximum bin number to fit.
- `LAST_C3`: The last C3 value to scan.
- `ENERGY`: The energy of the collision.
- `average_weight`: The average weight of the particles.

### Step 1: Prepare Input Data
```bash
# Prepare your 2D histogram data in ROOT format
# Place all data files into the ./rootfiles/ directory
```

### Step 2: Configure Centrality Binning
Edit the `centBin.py` file:
```python
# Locate the bin_ref3 array and set your centrality bin edges
bin_ref3 = [edge1, edge2, edge3, ...]
```

### Step 3: Generate Proton PDFs
```bash
sh FirstStep.sh
# This will generate proton PDFs for each ref3 bin
```

### Step 4: Calculate Cumulants for Each Centrality Bin
```bash
sh SecondStep.sh
# This will calculate cumulants from each centrality bin
```

### Step 5: Determine Average Factor
```bash
sh ThirdStep.sh
# This will calculate the average factor for subsequent analysis
```

### Step 6: Set Minimum Fitting Range (Low Limit C1)

1. Edit configuration files:
   - Input the average factor in `Run.py`
   - Set efficiency parameters in `Efficiency.py`

2. Run the test script:
   ```bash
   python3 test1.py
   # This will calculate the low limit C1
   ```

3. **Set Minimum Fitting Range**:
   - Determine the lower bound by identifying the intersection point where:
     - PDF from peripheral region's total distribution = 1e-5
     - Proton's total distribution PDF = 1e-5
   - This intersection point defines the minimum fitting range

### Step 7: Calculate low limit C1

1. Run the second test script:
   ```bash
   python3 test1.py
   # This will calculate the low limit C1
   ```

### Step 8: Scan Maximum Fitting Range and Determine Best C3 Value

1. Enable `ScanMax` in `Run.py`

2. Scan the maximum fitting range:
   ```bash
   bash ScanMax.sh
   # Fix the maximum fitting range based on results
   ```

3. Determine the best C3 value:
   ```bash
   bash ScanOnlyC3.sh
   # This will find the optimal central C3 value
   ```
   - Note: If volume fluctuation is not extreme, the C3 value should be around RefMult3's results
   - For large volume fluctuations, scan C3 value over a wider range

### Step 9: Generate Final Results

1. Input the best C3 value into `Run.py`

2. Enable `RunResults` in `Run.py`

3. Generate final results:
   ```bash
   python3 Run.py
   # This will produce the final analysis results
   ```

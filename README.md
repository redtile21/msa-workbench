# MSA Workbench

A desktop application for Measurement System Analysis (MSA).

## Installation

1. Create a virtual environment:
   ```shell
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```shell
     .\.venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```shell
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```shell
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Usage

Run the application with:
```shell
python -m msa_workbench
```

## Features

### MSA Analysis
1.  Go to the **MSA Analysis** tab.
2.  Click **Load CSV...** to load your measurement data. The application will automatically suggest a configuration for the response, factors, part, and operator based on common naming conventions.
3.  Review and override any of the suggested settings as needed.
4.  Choose the model type ("Crossed" or "Main Effects").
5.  Optionally, provide specification limits (LSL/USL) or a tolerance value.
6.  Click **Run Analysis**. The results will be displayed in the **Analysis Results** tab.

### MSA Builder
1.  Go to the **MSA Builder** tab.
2.  Enter 1 to 4 factor names and their corresponding levels (one level per line).
3.  Set the number of replicates for each factor combination.
4.  Choose an output order: "Left-to-right" (standard sorted order) or "Randomized".
    - For randomized order, you can optionally provide a seed for reproducibility.
5.  Click **Generate / Update Table** to see a preview.
6.  Click **Export CSV...** to save the generated run table.

## Building

To build an executable, use PyInstaller:
```shell
pyinstaller --noconfirm --clean --name msa-workbench --windowed -p src src/msa_workbench/__main__.py
```

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

## Building

To build an executable, use PyInstaller:
```shell
pyinstaller --noconfirm --clean --name msa-workbench --windowed -p src src/msa_workbench/__main__.py
```

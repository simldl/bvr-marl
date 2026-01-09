# Suggested Commands for Development

## Environment Setup

### Activate Conda Environment
```bash
conda activate rlenv
```

### Install/Update Dependencies
```bash
pip install -r requirements_gpu.txt
```

### Recreate Environment from Scratch
```bash
conda env create -f environment_gpu.yml
conda activate rlenv
```

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_speed_control.py
```

### Run Specific Test Function
```bash
pytest tests/test_speed_control.py::test_speed_control_initialization
```

### Run Tests with Verbose Output
```bash
pytest tests/ -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run Tests in Specific Directory
```bash
pytest tests/missiles/
pytest tests/aircrafts/
pytest tests/physics/
```

## Training

### Run RL Training
```bash
python reinforcement_learning/train.py
```

### Run Training with Custom Config
```bash
python reinforcement_learning/train.py --config-name custom_config
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir reinforcement_learning/logs
```

## Windows-Specific Commands

### List Directory Contents
```cmd
dir
# Or use PowerShell
ls
```

### Find Files
```cmd
dir /s /b *.py
# Or use PowerShell
Get-ChildItem -Recurse -Filter "*.py"
```

### Search File Contents (Windows)
```cmd
findstr /s /i "search_term" *.py
# Or use PowerShell
Select-String -Path *.py -Pattern "search_term" -Recurse
```

### Navigate Directories
```cmd
cd C:\Users\Simon\PythonProjects\air_to_air_rl
cd aircrafts
cd ..
```

### View File Contents
```cmd
type file.py
# Or use PowerShell
cat file.py
Get-Content file.py
```

### Python REPL
```bash
python
# Or
ipython  # if installed
```

## Git Commands

### Check Status
```bash
git status
```

### View Current Branch
```bash
git branch
```

### View Recent Commits
```bash
git log --oneline -10
```

### View Changes
```bash
git diff
```

### Stage and Commit
```bash
git add .
git commit -m "Your commit message"
```

### Switch Branches
```bash
git checkout main
git checkout -b new-feature-branch
```

### Pull Latest Changes
```bash
git pull origin main
```

## Development Workflow

### Typical Development Cycle
1. Activate environment: `conda activate rlenv`
2. Make code changes
3. Run relevant tests: `pytest tests/test_<feature>.py`
4. Run full test suite: `pytest tests/`
5. If training changes: `python reinforcement_learning/train.py`
6. Review results in TensorBoard
7. Commit changes: `git add . && git commit -m "Description"`

## Debugging and Analysis

### Run Python Script
```bash
python path/to/script.py
```

### Run with Python Debugger
```bash
python -m pdb path/to/script.py
```

### Check Python/Package Versions
```bash
python --version
pip list
conda list
```

### GPU Availability Check
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common File Operations

### Create Directory
```cmd
mkdir new_directory
```

### Copy Files
```cmd
copy source.py destination.py
xcopy /s /e source_dir dest_dir
```

### Delete Files/Directories
```cmd
del file.py
rmdir /s /q directory_name
```

## Notes for Windows Development

- Use forward slashes `/` or escaped backslashes `\\` in Python code for paths
- Path separator in Windows is `;` (not `:` like Unix)
- Line endings: Windows uses CRLF (`\r\n`), Git should handle this automatically
- PowerShell is more powerful than cmd.exe for scripting
- Consider using Windows Terminal for better CLI experience

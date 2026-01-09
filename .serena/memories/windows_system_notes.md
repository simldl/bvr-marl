# Windows System Notes

This project is developed on Windows. Here are important notes about Windows-specific behaviors and commands.

## Environment

- **OS**: Windows 10/11
- **Shell**: Command Prompt (cmd.exe) or PowerShell
- **Python Environment**: Conda (environment name: `rlenv`)
- **CUDA**: Version 12.8 for GPU acceleration

## Command Differences from Unix/Linux

### File System
- **Path separators**: Use backslash `\` in Windows (but Python accepts forward slash `/`)
- **Drive letters**: Paths start with drive letter, e.g., `C:\Users\Simon\...`
- **Case insensitive**: Filenames and paths are case-insensitive on Windows
- **Home directory**: `C:\Users\<username>\` (not `~`)

### Common Command Equivalents

| Unix/Linux | Windows CMD | PowerShell |
|------------|-------------|------------|
| `ls` | `dir` | `ls` or `dir` |
| `ls -la` | `dir /a` | `ls -Force` |
| `cat file.txt` | `type file.txt` | `cat file.txt` or `Get-Content` |
| `grep pattern` | `findstr pattern` | `Select-String -Pattern` |
| `find . -name "*.py"` | `dir /s /b *.py` | `Get-ChildItem -Recurse -Filter "*.py"` |
| `rm file` | `del file` | `Remove-Item file` |
| `rm -rf dir` | `rmdir /s /q dir` | `Remove-Item -Recurse -Force dir` |
| `cp src dst` | `copy src dst` | `Copy-Item src dst` |
| `mv src dst` | `move src dst` | `Move-Item src dst` |
| `pwd` | `cd` (no args) | `pwd` or `Get-Location` |
| `which python` | `where python` | `Get-Command python` |
| `clear` | `cls` | `cls` or `Clear-Host` |

### Environment Variables
- **View all**: `set` (CMD) or `Get-ChildItem Env:` (PowerShell)
- **Set variable**: `set VAR=value` (CMD) or `$env:VAR="value"` (PowerShell)
- **Path separator**: Semicolon `;` not colon `:`
- **Common paths**:
  - User home: `%USERPROFILE%` (CMD) or `$env:USERPROFILE` (PowerShell)
  - Temp directory: `%TEMP%` (CMD) or `$env:TEMP` (PowerShell)

## Conda on Windows

### Activation
```bash
conda activate rlenv
```

### Deactivation
```bash
conda deactivate
```

### Check Current Environment
```bash
conda env list
# or
conda info --envs
```

### Conda Prompt
Windows has a special "Anaconda Prompt" that pre-configures the conda environment.

## Python on Windows

### Running Python
```bash
python script.py
# or if multiple Python versions
py -3.12 script.py
```

### Python Launcher
Windows has `py` launcher that can select Python versions:
```bash
py --list          # List installed Python versions
py -3.12           # Run Python 3.12
py -3.12 -m pip    # Run pip for Python 3.12
```

## File Paths in Code

### Use Path Library (Recommended)
```python
from pathlib import Path

project_root = Path(__file__).parent.parent
data_file = project_root / "data" / "file.txt"
```

### Use os.path (Cross-platform)
```python
import os

path = os.path.join("dir", "subdir", "file.txt")
```

### Avoid Hardcoded Backslashes
```python
# Bad
path = "C:\Users\Simon\file.txt"  # Escape issues!

# Good
path = "C:/Users/Simon/file.txt"  # Works on Windows
path = r"C:\Users\Simon\file.txt"  # Raw string
path = Path("C:/Users/Simon/file.txt")  # Best
```

## Git on Windows

- **Line endings**: Git converts LF to CRLF automatically (configured in `.gitattributes` or `core.autocrlf`)
- **Git Bash**: Many developers use Git Bash for Unix-like shell on Windows
- **Credentials**: Windows Credential Manager stores Git credentials

### Check Line Ending Settings
```bash
git config --get core.autocrlf
# Should be "true" or "input" for Windows
```

## Performance Notes

### File I/O
- Windows file I/O can be slower than Linux, especially for many small files
- Antivirus scanning can slow down file operations

### Process Creation
- Creating subprocesses is slower on Windows
- Python multiprocessing may behave differently

### Long Paths
- Windows has 260-character path limit (MAX_PATH) by default
- Can be enabled for long paths in Windows 10+
- Use short directory names to avoid issues

## GPU/CUDA on Windows

### Check GPU
```bash
nvidia-smi
```

### Check CUDA in Python
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Common Issues
- **CUDA version mismatch**: PyTorch CUDA version must match installed CUDA toolkit
- **Driver updates**: Keep NVIDIA drivers up to date
- **Out of memory**: Windows reserves some GPU memory for display

## Terminal Recommendations

### Windows Terminal (Recommended)
- Modern terminal with tabs, Unicode support, theming
- Download from Microsoft Store
- Better than default Command Prompt

### PowerShell vs CMD
- **PowerShell**: More powerful, better scripting, Unix-like commands
- **CMD**: Simpler, legacy, faster startup
- Both work for basic development tasks

### Git Bash
- Unix-like shell on Windows
- Comes with Git for Windows
- Good for running Unix shell scripts

## Troubleshooting

### Permission Issues
- Run as Administrator if permission denied
- Check antivirus isn't blocking operations

### Path Too Long Errors
- Enable long path support in Windows settings
- Use shorter directory names
- Move project closer to root: `C:\projects\`

### Line Ending Issues
- Configure Git properly: `git config --global core.autocrlf true`
- Use `.gitattributes` file to enforce line endings

### Module Not Found
- Verify conda environment is activated
- Check `PYTHONPATH` if needed
- Reinstall requirements: `pip install -r requirements_gpu.txt`

## Best Practices

1. **Always activate conda environment** before running Python
2. **Use forward slashes** `/` in Python code (works on Windows)
3. **Use `pathlib.Path`** for path manipulation
4. **Keep paths short** to avoid MAX_PATH issues
5. **Check GPU utilization** with `nvidia-smi` during training
6. **Use Windows Terminal** or PowerShell for better experience
7. **Configure Git line endings** properly for team collaboration

---
description: How to deploy mlguide to PyPI and use it locally
---

# Deploying and Using mlguide

### 1. Local Use (Development Mode)
If you want to use the package on your computer and have changes you make to the code take effect immediately:

1. Open a terminal in the project root (`d:\joeinarthur\AutocleanML`).
2. Run:
   ```bash
   pip install -e .
   ```
3. You can now import it in any Python script on your system:
   ```python
   import autocleanml
   ```

### 2. Building the Package
To create the distribution files (`.whl` and `.tar.gz`):

1. Install the build tool:
   ```bash
   pip install build
   ```
2. Build the package:
   ```bash
   python -m build
   ```
   This will create a `dist/` folder with your package files.

### 3. Uploading to PyPI (Publishing)
To make your package available via `pip install autocleanml` for everyone:

1. Install Twine:
   ```bash
   pip install twine
   ```
2. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```
   *Note: You will need a PyPI account and an API token.*

### 4. Uploading to TestPyPI (Recommended first step)
To test the upload without making it public on the main PyPI:

1. Run:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
2. Test the installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ autocleanml
   ```

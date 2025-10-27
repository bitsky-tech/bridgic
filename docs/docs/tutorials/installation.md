# Installation

Bridgic is a next-generation agent development framework that enables developers to build agentic systems. Here are the installation instructions.

Python 3.9 or higher version is required.

=== "pip"

    ```bash
    pip install bridgic
    ```

=== "uv"

    ```bash
    uv add bridgic
    ```

After installation, you can verify that the installation was successful by running:

```bash
python -c "from bridgic.core import __version__; print(f'Bridgic version: {__version__}')"
```

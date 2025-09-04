Contributing to Bridgic
=======================

🎯 Quick Start Guide
--------------------

### Prerequisites Installation

We use uv as the package and project manager for all the Python packages in this repository. Before contributing, make sure you have uv installed.

On macOS and Linux:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official ducumentation](https://docs.astral.sh/uv/getting-started/installation/).


### Environment Preparation

First of all, you need to install the git hooks for this repository. To install the git hooks, simply run the following command in the project root directory:
```shell
make init-dev
```

Before developing, you have to navigate to the subpackage that you want to contribute to and set up its corresponding dependencies environment. For example, if you want to work on `bridgic-core`:
```shell
cd bridgic-core/
```

Create the virtual environment explicitly and activate it:
```shell
make venv-init
source .venv/bin/activate
```

Then you can run the tests in the current package that you're going to work on:
```shell
uv run -- pytest -s
```

**That's it!** The package you're going to work on is already installed in the editable mode and you have run the test cases for the first time. So you can go on, change the code and enjoy your journey in developing the Bridgic world.
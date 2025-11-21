Opik Observability Integration
==============================

Installation
-----

```shell
# Automatically install the Opik package
pip install bridgic-traces-opik
```

Prerequisites
-----

Before using `OpikTraceCallback`, you need to configure Opik. You can choose between two options:

- To use the hosted version, you need to [create a Comet account](https://www.comet.com/signup) and [grab your API Key](https://www.comet.com/account-settings/apiKeys).
- To run the Opik platform locally, see the [installation guide](https://www.comet.com/docs/opik/self-host/overview/) for more information.


The recommended way to configure the Python SDK is to run the `opik configure` command. It prompts you to enter your API key and, if applicable, the Opik instance URL so requests are routed and authenticated correctly. All details are saved to a configuration file.

If you are using the Cloud version of the platform, you can configure the SDK by running:

```python
import opik

opik.configure(use_local=False)
```

You can also configure the SDK by calling [`configure`](https://www.comet.com/docs/opik/python-sdk-reference/cli.html) from the command line:

```bash
opik configure
``` 

If you are self-hosting the platform, you can configure the SDK by running:

```python
import opik

opik.configure(use_local=True)
```

or from the command line:

```bash
opik configure --use_local
```

Both variants of `configure` prompt you for the required information and save it to `~/.opik.config`. When using the command-line version, you can pass the `-y` or `--yes` flag to automatically approve any confirmation prompts:

```bash
opik configure --yes
```

Once configured, you can start using `OpikTraceCallback` in your Bridgic applications.

Usage
-----

The `OpikTraceCallback` can be configured using the `start_opik_trace` function, which provides a convenient way to register Opik tracing for all workers application-wide.


The `start_opik_trace` function provides a convenient way to register Opik tracing for all workers application wide. This is the simplest and most recommended approach:

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.opik import start_opik_trace
import asyncio

# Register Opik trace callback
start_opik_trace(project_name="my-project")

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        return "hello"
    
    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
        return f"{step1} world"

async def main():
    automa = MyAutoma()  # Automatically uses global callback
    result = await automa.arun()
    print(result)

asyncio.run(main())
```

Parameters
----------

### `start_opik_trace` Parameters

- `project_name` (Optional[str]): The name of the project. If None, uses the `Default Project` project name.
- `workspace` (Optional[str]): The name of the workspace. If None, uses the `default` workspace name.
- `host` (Optional[str]): The host URL for the Opik server. If None, defaults to `https://www.comet.com/opik/api`.
- `api_key` (Optional[str]): The API key for Opik. This parameter is ignored for local installations.
- `use_local` (bool): Whether to use a local Opik server. Default is False.

Features
--------

- **Worker-level tracing**: Each worker execution is traced as a separate span
- **Nested automa support**: Properly handles nested automa instances with hierarchical tracing
- **Error tracking**: Captures and reports errors during worker execution
- **Execution metadata**: Tracks execution duration, start/end times, and other metadata
- **Concurrent execution**: Supports tracing multiple concurrent automa executions
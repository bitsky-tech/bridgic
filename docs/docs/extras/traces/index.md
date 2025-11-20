# Trace Integration


## Overview

Any agentic system needs a reliable observability layer to stay debuggable and trustworthy. The full picture of observability spans two complementary consists of these two parts:

- **Passive Tracing**: To Keep watching over every worker lifecycle event without code intrusion.
- **Active Tracing**: To enable application to emit custom spans or metrics whenever richer context is needed.

Together they form a holistic observability surface.

Bridgic fully supports passive tracing based on [Worker Callback Mechanism](../../tutorials/items/core_mechanism/worker_callback.ipynb), and provides seamless integration with several third-party platforms. More convient programming tools that support active tracing will also be introduced in the future.


## Available Integrations

| Package               | Description                                             |
|:----------------------|:--------------------------------------------------------|
| `bridgic-traces-opik` | Opik-powered observability and runtime tracing adapter. |
| `bridgic-traces-langwatch` | LangWatch-powered observability and runtime tracing adapter. |

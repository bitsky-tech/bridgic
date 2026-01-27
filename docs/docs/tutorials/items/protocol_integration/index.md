# Protocol Integration

## Introduction

Protocol integration modules in Bridgic enable seamless connecting with third-party protocols, allowing AI applications to leverage external tools, resources or services. This tutorial provides a basic overview of how to integrate with external protocols or SDKs and the benefits that will be brought to AI application development.

## The Value of Protocol Integration

Developing AI applications often requires interacting with various external services and resources. Because there are so many different platforms and systems, developers frequently need to work with specific protocols or SDKs for each one, which can increase the learning curve and add complexity to the development process.

In the world of Bridgic, every component of a workflow can exist as an fine-grained worker. Multiple workers can be orchestrated into a more complex automa object (typically in a dynamic graph form), which results in a bigger part of the complete AI system.

What if there were a design that turned the integration of third-party protocols or SDKs into simply using workers and orchestrating automas? This approach would significantly reduce developers' mental overhead while paving the way for smoother development, debugging, observability, and troubleshooting!

Bridgic addresses this need by:

- **Standard Neutrality**: Bridgic implements an integration layer tailored to its core architecture, adapting to the features and abstractions of each third-party protocol.
- **Unified Experience**: Developers use the same familiar Bridgic interfaces and patterns to access protocol integrations—no need to learn a separate API for each protocol.

## Available Integrations

Integrating external protocols will greatly expand the capabilities of your AI applications. Bridgic has already provided some important protocol integration packages, enabling you to unlock new possibilities.

| Protocol | Package | Description |
|:--------:|:-------:|:------------|
| [MCP](mcp_quick_start.ipynb) | `bridgic-protocols-mcp` | Model Context Protocol integration |

We warmly invite developers from across the community to join us in expanding Bridgic's protocol integrations. Your contribution will help shape a richer ecosystem, benefiting everyone building AI application more easily.

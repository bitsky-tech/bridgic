# Protocol Integration

## Overview

Bridgic's protocol integration module (`bridgic.protocols`) provides seamless integration with third-party protocols, enabling AI applications to leverage external tools, resources, and services through standardized interfaces.

## Design Philosophy

Bridgic integrates third-party protocols and SDKs through core abstractions tailored to its architecture, allowing developers to effortlessly use familiar development abilities and features.

- **Bridgic-Centric Abstractions**: Third-party protocols are integrated through Bridgic's core abstractions (like tools, prompts, workers), allowing developers to work with familiar Bridgic concepts
- **Seamless Development Experience**: Developers can use protocol tools and resources through the same patterns they already know from Bridgic, without needing to learn so many specific APIs

The protocol integration architecture is designed for extensibility:

- **Modular Design**: Each protocol has its own dedicated integration package
- **Future-Proof**: New protocol is not difficult to be addded

## Available Integrations

| Protocol | Package | Description |
|:--------:|:-------:|:------------|
| MCP | `bridgic-protocols-mcp` | Model Context Protocol integration |


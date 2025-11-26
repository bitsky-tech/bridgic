# 安装指南

Bridgic 是一个下一代代理开发框架，使开发者能够构建代理系统。以下是安装说明。

需要 Python 3.9 或更高版本。

=== "pip"

    ```bash
    pip install bridgic
    ```

=== "uv"

    ```bash
    uv add bridgic
    ```

安装后，您可以通过运行以下命令来验证安装是否成功：

```bash
python -c "from bridgic.core import __version__; print(f'Bridgic version: {__version__}')"
```
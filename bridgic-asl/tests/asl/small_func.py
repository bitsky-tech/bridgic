import inspect

def f(*args, **kwargs):
    pass

# 默认情况下：
print("默认签名：", inspect.signature(f))
# 输出：(*args, **kwargs)

# 动态伪造一个签名
sig = inspect.Signature([
    inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
    inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default="hi"),
])
f.__signature__ = sig

print("修改后签名：", inspect.signature(f))
# 输出：(x: int, y: str = 'hi')

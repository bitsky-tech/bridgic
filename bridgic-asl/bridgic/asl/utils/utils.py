from typing import List


class TokenIterator:
    """
    A iterator that iterates over a list of tokens. Consider two reasons:
    1. Because when ASLTranslator is executed in the Visitor pattern, it repeatedly enters the corresponding parsing function and cannot loop in place.
    2. Let the transformer focus on translating the syntax tree and not worry about other details.
    """
    def __init__(self, token_list: List):
        self.token_list = token_list[0][3]
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.token_list):
            token = self.token_list[self.index]
            self.index += 1
            return token
        
        # TODO: 这里如果越界，应该抛出异常，但理论上，这里不应该出现错误，否则是编译器本身出错了，得再多想想
        raise StopIteration
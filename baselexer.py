import re
from basetypes import BaseType

# Define token types
TOKEN_TYPES = {
    "KEYWORD": r"\b(pr|if|else|ty|io|ls|func|end)\b",
    "NUMBER": r"\b\d+(\.\d+)?\b",
    "STRING": r'".*?"|\'.*?\'',
    "IDENTIFIER": r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
    "OPERATOR": r"[=+\-*/<>!]+",
    "SEPARATOR": r"[,;:.]",
    "WHITESPACE": r"\s+",
}

class Token:
    def __init__(self, type_, value, line, column):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {self.value}, line={self.line}, column={self.column})"

class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.current_line = 1
        self.current_column = 1

    def tokenize(self):
        position = 0
        while position < len(self.source_code):
            match = None
            for token_type, regex in TOKEN_TYPES.items():
                pattern = re.compile(regex)
                match = pattern.match(self.source_code, position)
                if match:
                    value = match.group(0)
                    if token_type != "WHITESPACE":  # Ignore whitespace
                        self.tokens.append(Token(token_type, value, self.current_line, self.current_column))
                    self.current_column += len(value)
                    position += len(value)
                    break
            
            if not match:
                raise SyntaxError(f"Unexpected character '{self.source_code[position]}' at line {self.current_line}, column {self.current_column}")
        
        return self.tokens
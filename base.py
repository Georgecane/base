import os
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict
import jax.numpy as jnp  # For AI operations support with JAX

class TokenType(Enum):
    # Basic syntax tokens
    VAR = "var"
    PRINT = "pr"
    COLON = ":"
    SEMICOLON = ";"
    ASSIGN = "="
    
    # Data type tokens
    TYPE = "TYPE"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    
    # Value tokens
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int

class TypeSystem:
    def __init__(self):
        self.type_mapping = {
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'byte': bytes,
            'narray': jnp.ndarray,
            'tensor': jnp.ndarray,
            'vector': jnp.ndarray,
            'matrix': jnp.ndarray,
        }
    
    def register_type(self, base_type: str, python_type: Any):
        self.type_mapping[base_type] = python_type
    
    def get_python_type(self, base_type: str) -> Optional[Any]:
        return self.type_mapping.get(base_type)
    
    def is_valid_type(self, type_name: str) -> bool:
        return type_name in self.type_mapping

class Lexer:
    def __init__(self, filename: str):
        # Ensure the filename is a relative path and does not contain directory traversal
        if os.path.isabs(filename) or '..' in filename:
            raise ValueError("Invalid file path. Only relative paths without directory traversal are allowed.")
        
        base_dir = os.path.abspath('.')
        full_path = os.path.join(base_dir, filename)
        
        if not os.path.isfile(full_path):
            raise ValueError("Invalid file path. The file does not exist.")
        
        with open(full_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        
        self.pos = 0
        self.current_char = self.text[0] if self.text else None
        self.line = 1
        self.column = 1
        self.type_system = TypeSystem()
    
    def error(self):
        raise Exception(f'Invalid character at line {self.line}, column {self.column}')
    
    def advance(self):
        self.pos += 1
        self.column += 1
        
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            if self.text[self.pos] == '\n':
                self.line += 1
                self.column = 0
            self.current_char = self.text[self.pos]
    
    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def get_identifier(self) -> Token:
        result = ''
        column_start = self.column
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        if self.type_system.is_valid_type(result):
            return Token(TokenType.TYPE, result, self.line, column_start)
            
        if result == 'var':
            return Token(TokenType.VAR, result, self.line, column_start)
        
        if result == 'pr':
            return Token(TokenType.PRINT, result, self.line, column_start)
            
        return Token(TokenType.IDENTIFIER, result, self.line, column_start)
    
    def get_number(self) -> Token:
        result = ''
        column_start = self.column
        is_float = False
        
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if is_float:
                    self.error()
                is_float = True
            result += self.current_char
            self.advance()
        
        if is_float:
            return Token(TokenType.FLOAT, float(result), self.line, column_start)
        return Token(TokenType.INTEGER, int(result), self.line, column_start)
    
    def get_next_token(self) -> Token:
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isalpha() or self.current_char == '_':
                return self.get_identifier()
            
            if self.current_char.isdigit():
                return self.get_number()
            
            if self.current_char == ':':
                char = self.current_char
                self.advance()
                return Token(TokenType.COLON, char, self.line, self.column - 1)
                
            if self.current_char == '=':
                char = self.current_char
                self.advance()
                return Token(TokenType.ASSIGN, char, self.line, self.column - 1)
                
            if self.current_char == ';':
                char = self.current_char
                self.advance()
                return Token(TokenType.SEMICOLON, char, self.line, self.column - 1)
                
            if self.current_char == '[':
                char = self.current_char
                self.advance()
                return Token(TokenType.LBRACKET, char, self.line, self.column - 1)
                
            if self.current_char == ']':
                char = self.current_char
                self.advance()
                return Token(TokenType.RBRACKET, char, self.line, self.column - 1)
                
            if self.current_char == ',':
                char = self.current_char
                self.advance()
                return Token(TokenType.COMMA, char, self.line, self.column - 1)
            
            self.error()
        
        return Token(TokenType.EOF, None, self.line, self.column)

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
    
    def error(self):
        raise Exception(f'Invalid syntax at line {self.current_token.line}, column {self.current_token.column}')
    
    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()
    
    def parse(self):
        statements = []
        while self.current_token.type != TokenType.EOF:
            if self.current_token.type == TokenType.VAR:
                statements.append(self.parse_var_declaration())
            elif self.current_token.type == TokenType.PRINT:
                statements.append(self.parse_print_statement())
            else:
                self.error()
        return statements
    
    def parse_var_declaration(self):
        self.eat(TokenType.VAR)
        var_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.COLON)
        var_type = self.current_token.value
        self.eat(TokenType.TYPE)
        self.eat(TokenType.ASSIGN)
        var_value = self.parse_value(var_type)
        self.eat(TokenType.SEMICOLON)
        return ('var_declaration', var_name, var_type, var_value)
    
    def parse_print_statement(self):
        self.eat(TokenType.PRINT)
        var_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.SEMICOLON)
        return ('print_statement', var_name)
    
    def parse_value(self, var_type: str):
        if var_type == 'int':
            value = self.current_token.value
            self.eat(TokenType.INTEGER)
            return int(value)
        elif var_type == 'float':
            value = self.current_token.value
            self.eat(TokenType.FLOAT)
            return float(value)
        elif var_type == 'str':
            value = self.current_token.value
            self.eat(TokenType.STRING)
            return str(value)
        elif var_type == 'bool':
            value = self.current_token.value
            self.eat(TokenType.IDENTIFIER)
            return value.lower() == 'true'
        elif var_type == 'byte':
            value = self.current_token.value
            self.eat(TokenType.STRING)
            return bytes(value, 'utf-8')
        elif var_type in ['narray', 'vector', 'matrix', 'tensor']:
            values = []
            self.eat(TokenType.LBRACKET)
            while self.current_token.type != TokenType.RBRACKET:
                values.append(self.current_token.value)
                self.eat(TokenType.INTEGER)
                if self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
            self.eat(TokenType.RBRACKET)
            values = jnp.array(values)
            if var_type == 'narray':
                return values
            elif var_type == 'vector':
                return values
            elif var_type == 'matrix':
                return values.reshape(-1, len(values))
            elif var_type == 'tensor':
                return values
        else:
            self.error()

class Interpreter:
    def __init__(self, parser: Parser):
        self.parser = parser
        self.variables = {}
    
    def interpret(self):
        statements = self.parser.parse()
        for statement in statements:
            if statement[0] == 'var_declaration':
                _, var_name, var_type, var_value = statement
                self.variables[var_name] = var_value
            elif statement[0] == 'print_statement':
                _, var_name = statement
                if var_name in self.variables:
                    print(f"{var_name} = {self.variables[var_name]}")
                else:
                    print(f"Error: Variable {var_name} not defined")

def main():
    parser = argparse.ArgumentParser(description="Run the Base language interpreter.")
    parser.add_argument('filename', metavar='F', type=str, help='the file to interpret')
    args = parser.parse_args()
    
    lexer = Lexer(args.filename)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.interpret()

if __name__ == "__main__":
    main()

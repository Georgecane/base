from baselexer import Lexer, Token

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tokens = lexer.tokenize()
        self.current_token_index = 0
        self.current_token = self.tokens[self.current_token_index]

    def error(self, message):
        raise SyntaxError(f"Parsing Error: {message}")

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token_index += 1
            if self.current_token_index < len(self.tokens):
                self.current_token = self.tokens[self.current_token_index]
        else:
            self.error(f"Expected token {token_type}, but got {self.current_token.type}")

    def parse_statement(self):
        if self.current_token.type == "IDENTIFIER":
            return self.parse_assignment()
        elif self.current_token.type == "KEYWORD":
            if self.current_token.value == "pr":
                return self.parse_print()
            elif self.current_token.value == "if":
                return self.parse_if()
            elif self.current_token.value == "ty":
                return self.parse_type()
            elif self.current_token.value == "io":
                return self.parse_input()
            elif self.current_token.value == "func":
                return self.parse_func()
            elif self.current_token.value == "ls":
                return {'type': 'LS'}
        else:
            self.error(f"Unexpected token: {self.current_token.type}")

    def parse_assignment(self):
        var_name = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("OPERATOR")
        value = self.current_token.value
        self.eat("NUMBER")
        self.eat("SEPARATOR")
        return {'type': 'ASSIGN', 'name': var_name, 'value': value}

    def parse_print(self):
        self.eat("KEYWORD")
        expr = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("SEPARATOR")
        return {'type': 'PRINT', 'value': expr}

    def parse_if(self):
        self.eat("KEYWORD")
        condition = self.current_token.value
        self.eat("OPERATOR")
        self.eat("NUMBER")
        self.eat("SEPARATOR")
        statement = self.parse_statement()
        if self.current_token.value == "else":
            self.eat("KEYWORD")
            else_statement = self.parse_statement()
            self.eat("SEPARATOR")
            return {'type': 'IF', 'condition': condition, 'body': statement, 'else': else_statement}
        self.eat("SEPARATOR")
        return {'type': 'IF', 'condition': condition, 'body': statement}

    def parse_type(self):
        self.eat("KEYWORD")
        var_name = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("SEPARATOR")
        return {'type': 'TYPE', 'name': var_name}

    def parse_input(self):
        self.eat("KEYWORD")
        prompt = self.current_token.value
        self.eat("STRING")
        self.eat("SEPARATOR")
        return {'type': 'INPUT', 'prompt': prompt}

    def parse_func(self):
        self.eat("KEYWORD")
        func_name = self.current_token.value
        self.eat("IDENTIFIER")
        self.eat("OPERATOR")
        params = self.current_token.value.split(',')
        self.eat("SEPARATOR")
        body = []
        while self.current_token.value != "end":
            body.append(self.parse_statement())
        self.eat("KEYWORD")
        return {'type': 'FUNC', 'name': func_name, 'params': params, 'body': body}

    def parse(self):
        statements = []
        while self.current_token_index < len(self.tokens):
            statements.append(self.parse_statement())
        return statements
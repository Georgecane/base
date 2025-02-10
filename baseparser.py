from baselexer import Lexer, TokenType

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self, message):
        raise SyntaxError(f"Parsing Error: {message}")

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(f"Expected token {token_type}, but got {self.current_token.type}")

    def parse_statement(self):
        """Parses a single statement."""
        if self.current_token.type == TokenType.PRINT:
            return self.parse_print()
        elif self.current_token.type == TokenType.IF:
            return self.parse_if()
        elif self.current_token.type == TokenType.TYPE:
            return self.parse_type()
        elif self.current_token.type == TokenType.INPUT:
            return self.parse_input()
        elif self.current_token.type == TokenType.FUNC:
            return self.parse_func()
        elif self.current_token.type == TokenType.LS:
            return {'type': 'LS'}
        else:
            self.error(f"Unexpected token: {self.current_token.type}")

    def parse_print(self):
        """Parses a print statement."""
        self.eat(TokenType.PRINT)
        expr = self.current_token.value
        self.eat(TokenType.STRING)
        self.eat(TokenType.SEMICOLON)
        return {'type': 'PRINT', 'value': expr}

    def parse_if(self):
        """Parses an if statement."""
        self.eat(TokenType.IF)
        condition = self.current_token.value
        self.eat(TokenType.CONDITION)
        self.eat(TokenType.DO)
        statement = self.parse_statement()
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_statement = self.parse_statement()
            return {'type': 'IF', 'condition': condition, 'body': statement, 'else': else_statement}
        return {'type': 'IF', 'condition': condition, 'body': statement}
    
    def parse_type(self):
        """Parses a type statement."""
        self.eat(TokenType.TYPE)
        value = self.current_token.value
        self.eat(TokenType.NUMBER)
        self.eat(TokenType.SEMICOLON)
        return {'type': 'TYPE', 'value': value}

    def parse_input(self):
        """Parses an input statement."""
        self.eat(TokenType.INPUT)
        prompt = self.current_token.value
        self.eat(TokenType.STRING)
        self.eat(TokenType.SEMICOLON)
        return {'type': 'INPUT', 'prompt': prompt}
    
    def parse_func(self):
        """Parses a function definition."""
        self.eat(TokenType.FUNC)
        func_name = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.PARAMS)
        params = self.current_token.value.split(',')
        self.eat(TokenType.COLON)
        body = []
        while self.current_token.type != TokenType.END:
            body.append(self.parse_statement())
        self.eat(TokenType.END)
        return {'type': 'FUNC', 'name': func_name, 'params': params, 'body': body}
    
    def parse(self):
        """Parses a full program."""
        statements = []
        while self.current_token.type != TokenType.EOF:
            statements.append(self.parse_statement())
        return statements

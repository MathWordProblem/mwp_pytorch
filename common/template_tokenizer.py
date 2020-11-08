import re


mixed_num_pattern = re.compile('(\d+)\((\d+/\d+)\)')
mixed_equation_pattern = re.compile('(\d+)\(')
fraction_pattern = re.compile('\((\d+/\d+)\)')
percentage_pattern = re.compile('([\.\d]+)%')


class TemplateTokenizer(object):

    VARIABLE_TOKENS = ['δ', 'ε', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ο', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ']
    CONSTANT_TOKENS = {
        '1': 'б',
        '2': 'в',
        '3.14': 'π',
        '3': 'г',
        '4': 'г',
        '6': 'к',
        '7': 'д',
        '10': 'ж',
        '12': 'л',
        '30': 'з',
        '31': 'и',
        '60': 'ω',
        '100': 'ц',
        '1000': 'ш',
        '10000': 'ы'
    }
    NUMBER_PATTERN = re.compile(r'\-?\d{1,10}(?:\.\d{1,10})?')

    def __init__(self):
        # 键是一个问题的id，值是一个var_map
        # var_map的键是题目中的数字变量，值是替换后的符号
        self._variable_store = {}

    def _has_digits(self, text):
        for c in text:
            if '0' <= c <= '9':
                return False
        return True

    def build_templates(self, problem_id, problem_text):
        """Build variable map from problem text, and store with key of problem_id.
        
        """
        variables = []
        var_map = {}
        replaced_text = ''
        last = 0
        for m in re.finditer(self.NUMBER_PATTERN, problem_text):
            num = m.group(0)
            try:
                index = variables.index(num)
            except ValueError:
                index = len(variables)
                variables.append(num)
            token = self.VARIABLE_TOKENS[index]
            var_map[num] = token
            replaced_text += problem_text[last: m.start()] + token
            last = m.end()
        replaced_text += problem_text[last:]
        self._variable_store[problem_id] = var_map
        return replaced_text

    def transfer_equation(self, problem_id, equation):
        var_map = self._variable_store[problem_id]
        tokens = equation.split('|')
        new_tokens = []
        for token in tokens:
            if token in var_map.keys():
                token = var_map[token]
            elif token in self.CONSTANT_TOKENS.keys():
                token = self.CONSTANT_TOKENS[token]
            new_tokens.append(token)
        text = '|'.join(new_tokens)
        has_constant = self._has_digits(text)
        return text, has_constant

    def restore(self, problem_id, text):
        """Restore the token to original number.
        
        """
        # 变量恢复
        var_map = self._variable_store[problem_id]
        inv_map = {v: k for k, v in var_map.items()}
        inv_const = {v: k for k, v in self.CONSTANT_TOKENS.items()}
        new_tokens = []
        for c in text:
            if c in self.VARIABLE_TOKENS:
                c = inv_map[c]
            elif c in inv_const.keys():
                c = inv_const[c]
            new_tokens.append(c)
        text = ''.join(new_tokens)

        return text


import re

from common.utils import to_prefix, to_infix, is_equal


mixed_num_pattern = re.compile('(\d+)\((\d+/\d+)\)')
mixed_equation_pattern = re.compile('(\d+)\(')
fraction_pattern = re.compile('\((\d+/\d+)\)')
percentage_pattern = re.compile('([\.\d]+)%')

def preprocess(question, equation, answer):
    # 处理带分数
    question = re.sub(mixed_num_pattern, '(\\1+\\2)', question)
    equation = re.sub(mixed_num_pattern, '(\\1+\\2)', equation)
    answer = re.sub(mixed_num_pattern, '(\\1+\\2)', answer)
    equation = re.sub(mixed_equation_pattern, '\\1+(', equation)
    answer = re.sub(mixed_equation_pattern, '\\1+(', answer)
    # 分数去括号
    question = re.sub(fraction_pattern, '\\1', question)
    # 处理百分数
    equation = re.sub(percentage_pattern, '(\\1/100)', equation)
    answer = re.sub(percentage_pattern, '(\\1/100)', answer)
    # 冒号转除号、剩余百分号处理
    equation = equation.replace(':', '/').replace('%', '/100')
    answer = answer.replace(':', '/').replace('%', '/100')
    if equation[:2] == 'x=':
        equation = equation[2:]
    return question, equation, answer


def convert_to_prefix(question, equation, answer):
    # 把乘方符号从**换成^
    question = question.replace('**', '^')
    equation = equation.replace('**', '^')
    # 中缀表达式转前缀
    equation = to_prefix(equation)
    # 测试一下转换是不是有问题，能不能逆转
    temp = to_infix(equation)
    temp = temp.replace('^', '**')
    assert is_equal(eval(temp), eval(answer)), "Prefix template transfer error!"
    return question, equation, answer

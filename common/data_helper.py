import re
import json

from common.utils import to_prefix, to_infix
from common.utils import is_equal


def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    for l in open(filename):
        l = json.loads(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 分数去括号
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            if is_equal(eval(equation), eval(answer)):
                # 把乘方符号从**换成^
                question = question.replace('**', '^')
                equation = equation.replace('**', '^')
                # 中缀表达式转前缀
                equation = to_prefix(equation)
                # 测试一下转换是不是有问题，能不能逆转
                temp = to_infix(equation)
                temp = temp.replace('^', '**')
                assert(is_equal(eval(temp), eval(answer)))
                D.append((question, equation, answer))
        except BaseException as e:
            continue
    return D


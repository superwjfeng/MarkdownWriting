'''
Formatter for MarkdownWriting
'''

import re
import os

def insert_spaces(text):
    # 定义中文字符的 Unicode 范围
    chinese_char_pattern = r'[\u4e00-\u9fff]'
    
    # 为中文和数字之间插入空格
    text = re.sub(f"({chinese_char_pattern})(\d)", r"\1 \2", text)
    text = re.sub(f"(\d)({chinese_char_pattern})", r"\1 \2", text)

    # 为中文和英文之间插入空格
    text = re.sub(f"({chinese_char_pattern})([a-zA-Z])", r"\1 \2", text)
    text = re.sub(f"([a-zA-Z])({chinese_char_pattern})", r"\1 \2", text)
    
    # 为中文和 `` 或 $$ 之间的代码块插入空格
    text = re.sub(f"({chinese_char_pattern})(`+|\$\$)", r"\1 \2", text)
    text = re.sub(f"(`+|\$\$)({chinese_char_pattern})", r"\1 \2", text)

    return text

insert_spaces(os.path.join(os.getcwd(), 'CS', 'Coding', 'Cpp', 'Cpp联邦', '模板'))
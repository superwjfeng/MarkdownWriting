import re
import os

def format_text(text):
    # 排除<img>标签中的src属性内容
    def skip_img_tags(match):
        return match.group(0)
    
    img_pattern = re.compile(r'<img\s+[^>]*src="[^"]*"[^>]*>')
    text = re.sub(img_pattern, skip_img_tags, text)

    # 正则表达式匹配汉字与英文字母或数字之间以及反引号或美元符号的代码块之间没有空格的地方
    pattern = r'([\u4e00-\u9fff])([A-Za-z0-9`$])|([A-Za-z0-9`$])([\u4e00-\u9fff])'
    repl_func = lambda m: m.group(1) + ' ' + m.group(2) if m.group(2) else m.group(3) + ' ' + m.group(4)
    text_with_spaces = re.sub(pattern, repl_func, text)

    # 特殊处理中文与``或$$围起来的代码块之间的空格
    code_block_pattern = r'([\u4e00-\u9fff])(`{2,}.*?`{2,}|\${2,}.*?\${2,})|(`{2,}.*?`{2,}|\${2,}.*?\${2,})([\u4e00-\u9fff])'
    code_block_repl_func = lambda m: m.group(1) + ' ' + m.group(2) if m.group(1) else m.group(3) + ' ' + m.group(4)
    formatted_text = re.sub(code_block_pattern, code_block_repl_func, text_with_spaces)

    return formatted_text

def process_markdown_file(file_path):
    # 读取 markdown 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 格式化文本
    formatted_content = format_text(content)

    # 输出结果或写回到文件中
    # print(formatted_content) # 打印到控制台
    # 如果需要写回到文件，取消下面注释的代码行并注释掉上面的print语句。
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(formatted_content)

if __name__ == "__main__":
    markdown_file_path = os.path.join(os.getcwd(), 'CS', 'Coding', 'Cpp', 'Cpp联邦', '模板', '模板.md') # 替换为你的 Markdown 文件名
    process_markdown_file(markdown_file_path)

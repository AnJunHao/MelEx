import re

pattern = r"[a-zA-Z0-9-_\.]+@(?:[a-zA-Z0-9]+\.)+(?:com|edu)"
string = """anjunhao_23@163.com
anjunhao.18@gmail.com
junhaoa@andrew.cmu.edu
rae-yutong@outlook.com
wholovesyoumost?@claude.haha
asdefuh.2eugde.kk
huhuhuhu@loveyou
you are @ my.heart"""

print(re.findall(pattern, string))

# matches = list(re.finditer(pattern, string))
# for match in matches:
#     print(match.group())
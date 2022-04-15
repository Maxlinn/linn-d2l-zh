import os
import re
from urllib.parse import urlparse

## 递归遍历根目录下的.md和.ipynb文件，找出其中链到本地文件的[]()结构，并将其改成由
## docsify能解析同目录下的文件夹(assets/img.png)，但是不能解析相对于父目录的引用
## 而d2l-ai经常使用父目录引用，不改会显示不出来图片

base = os.getcwd()

# 默认DFS
for root, dirs, files in os.walk('.'):
    # root 是当前文件夹的全路径
    if os.path.basename(root).startswith('.'): 
        continue
    for fname in files:
        if not fname.endswith('.md') or fname.endswith('.ipynb'):
            continue
        filename = os.path.join(root, fname)
        filedirname = os.path.dirname(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        def replace_callback(matchobj):
            # 0 是源字符串
            text = matchobj.group(1)
            link = matchobj.group(2)
            if urlparse(link).scheme:
                return

            point_fname = os.path.join(filedirname, link)
            point_filename = os.path.realpath(point_fname)
            target_relfname = os.path.relpath(point_filename, base)
            target_relfname = target_relfname.replace('\\', '/')
            res = f'[{text}]({target_relfname})'
            return res
        content = re.sub(r'\[(.+?)\]\((.+?)\)',
                         replace_callback,
                         content)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)





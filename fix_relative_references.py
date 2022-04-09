import os
import re
from urllib.parse import urlparse

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





from typing import List
import os


def filter_ext(lz, ext='.md'):
    return list(filter(lambda fname: os.path.splitext(fname)[1] == ext, lz))


def filter_dir(lz):
    return list(filter(lambda fname: os.path.isdir(fname), lz))


# [(text, url), [children])
# 先处理本层，再处理下层

base = os.getcwd()


# def process_level(dst: List):
#     root = os.listdir()
#     toc_ = []
#     dirs = filter_dir(root)
#     dirs = list(filter(lambda fname: not fname.startswith('.'), dirs))
#     mds = filter_ext(root, '.md')
#     mds = list(filter(lambda fname: not fname.startswith('_'), mds))
#     ipynbs = filter_ext(root, '.ipynb')
#
#     for md in mds:
#         md_noext = os.path.splitext(md)[0]
#         md_noext_abs = os.path.abspath(md)
#         md_rel = os.path.relpath(md_noext_abs, base)
#         toc_.append((md_noext, md_rel))
#
#     for ipynb in ipynbs:
#         ipynb_noext = os.path.splitext(ipynb)[0]
#         ipynb_noext_abs = os.path.abspath(ipynb_noext)
#         ipynb_rel = os.path.relpath(ipynb_noext_abs, base)
#         toc_.append((ipynb_noext, ipynb_rel))
#
#     for dir in dirs:
#         cwd = os.getcwd()
#         os.chdir(dir)
#         toc_2 = []
#         process_level(toc_2)
#         os.chdir(cwd)
#         toc_.append(toc_2)
#     dst.append(toc_)


def process_level(dst: List):
    root = os.listdir()
    dirs = filter_dir(root)
    dirs = list(filter(lambda fname: not fname.startswith('.'), dirs))
    mds = filter_ext(root, '.md')
    mds = list(filter(lambda fname: not fname.startswith('_'), mds))
    ipynbs = filter_ext(root, '.ipynb')

    for md in mds:
        md_noext = os.path.splitext(md)[0]
        md_noext_abs = os.path.abspath(md)
        md_rel = os.path.relpath(md_noext_abs, base)
        dst.append((md_noext, md_rel))

    for ipynb in ipynbs:
        ipynb_noext = os.path.splitext(ipynb)[0]
        ipynb_noext_abs = os.path.abspath(ipynb_noext)
        ipynb_rel = os.path.relpath(ipynb_noext_abs, base)
        dst.append((ipynb_noext, ipynb_rel))

    for dir in dirs:
        cwd = os.getcwd()
        os.chdir(dir)
        process_level(dst)
        os.chdir(cwd)


toc = []
process_level(toc)

d = {}
for item in toc:
    text, link = item
    dirname, _ = os.path.split(link)
    root = dirname.split('\\')[0]
    # d[root] = f"- [{text}]({link})"
    d[root] = d.get(root, []) + [item]

f = open('_sidebar.md', 'w')


def to_str(tp): return f'- [{tp[0]}]({tp[1]})'


for root, lz in d.items():
    if root == '':
        for item in lz:
            f.write(f'{to_str(item)}\n')
    else:
        f.write(f'- {root}\n')
        for item in lz:
            f.write(f'\t{to_str(item)}\n')


f.close()

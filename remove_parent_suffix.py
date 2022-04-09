import os

for root, dirs, files in os.walk('.'):
    if os.path.basename(root).startswith('.'): continue
    for file in files:
        if not file.endswith('.ipynb'): continue
        file_abs = os.path.join(root, file)
        print(f'dealing {file_abs}')
        f = open(file_abs, 'r', encoding='utf-8')
        s = f.read()
        f.close()
        s = s.replace(r'../img', 'img')
        f = open(file_abs, 'w', encoding='utf-8')
        f.write(s)
        f.close()
        
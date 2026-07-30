import os
import glob
import numpy as np

files = glob.glob('host_software/**/*.py', recursive=True)
count = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        continue
    
    if 'MAX_X_BOUND' not in content:
        continue
        
    print(f'Patching {f}...')
    content = content.replace('MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0', 'MAX_X_BOUND, MAX_Y_BOUND = 200.0, 200.0')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
    count += 1
    
print(f'Patched {count} files.')

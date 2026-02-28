import os
import re

def clean_locks(directory):
    count = 0
    read_unwrap = re.compile(r'\.read\(\)\.unwrap\(\)')
    write_unwrap = re.compile(r'\.write\(\)\.unwrap\(\)')
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.rs'): continue
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
                
            new_content = read_unwrap.sub('.read()', content)
            new_content = write_unwrap.sub('.write()', new_content)
            
            if new_content != content:
                with open(path, 'w') as f:
                    f.write(new_content)
                count += 1
    print(f"Cleaned {count} files.")

if __name__ == "__main__":
    clean_locks("src")

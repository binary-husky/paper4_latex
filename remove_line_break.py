import re

def main(in_file):
    # run 1
    # run 1
    # run 1
    with open(in_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    lines = [line.strip(' ') for line in lines]

    for i, line in enumerate(lines):
        # remove latex comment
        if in_file.endswith('.tex'):
            lines[i] = lines[i].split('%', 1)[0]
        # such as \section{xxxx}
        if lines[i].endswith('}\n'):
            continue
        # empty line, ignore
        if lines[i].startswith('\n'):
            continue
        # remove line breaks
        if i+1 >= len(lines):
            break
        if not lines[i+1].startswith('\n'):
            lines[i] = lines[i].replace('\n', ' ')


    with open(in_file+'.nobreak.md', 'w') as f:
        f.writelines(lines)


    # run 2
    # run 2
    # run 2
    in_file = in_file+'.nobreak.md'
    with open(in_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip(' ') for line in lines]

    def remove_eq(line):
        return re.sub(r'\$\$(.*?)\$\$', r'EQUATION', line)
        
    def remove_eq_s(line):
        return re.sub(r'\$(.*?)\$', r'EQ', line)

    for i, line in enumerate(lines):
        # empty line, ignore
        if lines[i].startswith('\n'):
            continue
        # remove equation
        if '$$' in lines[i]:
            lines[i] = remove_eq(lines[i])
        # remove equation
        if '$' in lines[i]:
            lines[i] = remove_eq_s(lines[i])
        # replace textbf with its content
        lines[i] = re.sub(r'\\textbf{(.*?)}', r'\1', lines[i])
        lines[i] = re.sub(r'\\section{(.*?)}', r'\1', lines[i])
        lines[i] = re.sub(r'\\subsection{(.*?)}', r'\1', lines[i])
        lines[i] = re.sub(r'\\subsubsection{(.*?)}', r'\1', lines[i])
        lines[i] = re.sub(r'~\\ref{(.*?)}', r'', lines[i])
        lines[i] = re.sub(r'.\\ref{(.*?)}', r'', lines[i])
        lines[i] = re.sub(r'\\label{(.*?)}', r'', lines[i])
        lines[i] = re.sub(r'\\cite{(.*?)}', r'(CITE)', lines[i])
        # remove extra space
        lines[i] = lines[i].replace('  ', ' ')

    with open(in_file, 'w') as f:
        f.writelines(lines)


in_file = [
    "tex/00-abs.tex",
    "tex/01-intro.tex",

]

for f in in_file:
    main(f)




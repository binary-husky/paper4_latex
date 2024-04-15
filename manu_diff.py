import os, shutil, stat, fnmatch, time
reverse_commit = '5d70e424c85bceffcf9eb863bf9a3e8a3b38698a'  
main_file = 'submit_to_TFS'
input('确认已经git commit?')
time_mark = time.strftime("%m-%d", time.localtime())
master_branch_name = 'main'

def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)      

try: rmtree('./manu_diff')
except: pass
os.makedirs('./manu_diff')

name = 'old'
res = os.system(f'git checkout  -f {reverse_commit}')
assert res == 0
def ignore_fn(src, names):
    print(f'copy {src}')
    exclude = ['.git','manu_diff','trash','*.log','*.bbl','*(busy)*', "*.aux", "*.blg", "*.fdb_latexmk", "*.fls", "*.ai", "*.synctex.gz"]
    ignored_names = []
    for k in names:
        ignore = False
        for n in exclude:
            if fnmatch.fnmatch(k, n): 
                ignore = True
                break
        if ignore: 
            ignored_names.append(k)
    return ignored_names
shutil.copytree('./', f'./manu_diff/{name}', ignore=ignore_fn)

reverse_commit = master_branch_name
name = 'new'
os.system(f'git checkout -f {reverse_commit}')
shutil.copytree('./', f'./manu_diff/{name}', ignore=ignore_fn)

name = 'diff'
shutil.copytree('./', f'./manu_diff/{name}', ignore=ignore_fn)

shutil.copy(f'./manu_diff/new/{main_file}.tex', f'./manu_diff/old/{main_file}.tex')

input('inseart files finished, press enter')


os.chdir('./manu_diff/old'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
os.chdir('./manu_diff/new'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
os.chdir('./manu_diff/old'); os.system(f'bibtex {main_file}.aux'); os.chdir('../../')
os.chdir('./manu_diff/new'); os.system(f'bibtex {main_file}.aux'); os.chdir('../../')
os.chdir('./manu_diff/old'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
os.chdir('./manu_diff/new'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
os.chdir('./manu_diff/old'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
os.chdir('./manu_diff/new'); os.system(f'pdflatex {main_file}.tex'); os.chdir('../../')
# latexdiff --append-safecmd=subfile ./manu_diff/old/annoy_commit.tex  ./manu_diff/new/annoy_commit.tex --flatten > ./manu_diff/diff/000_diff.tex


print(    f'latexdiff --encoding=utf8 --append-safecmd=subfile ./manu_diff/old/{main_file}.tex  ./manu_diff/new/{main_file}.tex --flatten > ./manu_diff/diff/0000000_diff.tex')
os.system(f'latexdiff --encoding=utf8 --append-safecmd=subfile ./manu_diff/old/{main_file}.tex  ./manu_diff/new/{main_file}.tex --flatten > ./manu_diff/diff/0000000_diff.tex')

os.chdir('./manu_diff/diff'); os.system(f'pdflatex 0000000_diff.tex'); os.chdir('../../')
os.chdir('./manu_diff/diff'); os.system(f'bibtex 0000000_diff.tex'); os.chdir('../../')
os.chdir('./manu_diff/diff'); os.system(f'pdflatex 0000000_diff.tex'); os.chdir('../../')
os.chdir('./manu_diff/diff'); os.system(f'pdflatex 0000000_diff.tex'); os.chdir('../../')

shutil.copy(f'./manu_diff/diff/0000000_diff.pdf', f'./付清旭_{main_file}_{time_mark}_修改位置高亮.pdf')
shutil.copy(f'./manu_diff/new/{main_file}.pdf',   f'./付清旭_{main_file}_{time_mark}_修改后.pdf')

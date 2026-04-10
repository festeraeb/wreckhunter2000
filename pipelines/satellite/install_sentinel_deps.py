import subprocess
import sys

pkgs = ['pystac-client', 'planetary-computer', 'rasterio']

def run(cmd):
    print('RUN:', ' '.join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    return p.returncode

if __name__ == '__main__':
    rc = run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    if rc != 0:
        print('Failed to upgrade pip, continuing')
    rc = run([sys.executable, '-m', 'pip', 'install'] + pkgs)
    if rc == 0:
        print('Install complete')
    else:
        print('Install finished with errors; check output')

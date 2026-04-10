import importlib

def check(pkgs):
    missing = []
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except Exception as e:
            missing.append(f"{pkg}: {e}")
    return missing

if __name__ == '__main__':
    pkgs = ['pystac_client', 'planetary_computer', 'rasterio']
    missing = check(pkgs)
    if not missing:
        print('OK: all packages available')
    else:
        print('MISSING:')
        for m in missing:
            print('-', m)

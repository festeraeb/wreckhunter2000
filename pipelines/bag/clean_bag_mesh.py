import re

with open('src/bag_mesh.rs', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "Hdf5(" in line:
        continue
    if '#[cfg(feature = "hdf5")]' in line:
        skip = True
        continue
    if skip and line.startswith('}'):
        skip = False
        continue
    if skip:
        continue
    new_lines.append(line)

content = "".join(new_lines)
with open('src/bag_mesh.rs', 'w', encoding='utf-8') as f:
    f.write(content)

import json
import re
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 读取 requirements.txt 文件
with open(ROOT / 'requirements.txt', 'r') as file:
    requirements = file.readlines()

# 解析 requirements.txt 文件并构建 JSON 结构
packages = []
for requirement in requirements:
    # 忽略注释掉的内容
    if requirement.strip().startswith("#"):
        continue

    # 使用正则表达式匹配包名和版本号
    match = re.match(r'^([a-zA-Z0-9_-]+)==(\d+\.\d+\.\d+(?:\.\d+)?)', requirement.strip())
    if match:
        package_name = match.group(1)
        package_version = match.group(2)
        packages.append({
            "package_name": package_name,
            "package_version": package_version,
            "restraint": "EXACT"
        })

    match = re.match(r'^([a-zA-Z0-9_-]+)>=(\d+\.\d+\.\d+(?:\.\d+)?)', requirement.strip())
    if match:
        package_name = match.group(1)
        package_version = match.group(2)
        packages.append({
            "package_name": package_name,
            "package_version": package_version,
            "restraint": "ATLEAST"
        })

    match = re.match(r'^([a-zA-Z0-9_-]+)<=(\d+\.\d+\.\d+(?:\.\d+)?)', requirement.strip())
    if match:
        package_name = match.group(1)
        package_version = match.group(2)
        packages.append({
            "package_name": package_name,
            "package_version": package_version,
            "restraint": "ATMOST"
        })

    match = re.match(r'^([a-zA-Z0-9_-]+)', requirement.strip())
    if match:
        package_name = match.group(1)
        packages.append({
            "package_name": package_name,
        })

# 将构建的 JSON 结构转换为字符串
json_output = json.dumps({"packages": packages}, indent=4)

# 打印或保存 JSON 输出
print(json_output)

# 如果需要保存到文件
with open(ROOT / 'requirements.json', 'w') as f:
    f.write(json_output)
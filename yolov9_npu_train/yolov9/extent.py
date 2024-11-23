import os

# 定义函数，处理单个txt文件
def expand_bounding_boxes(txt_file_path, expansion_pixels=2):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    
    expanded_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Warning: Invalid line format in {txt_file_path}: {line.strip()}")
            expanded_lines.append(line.strip())
            continue
        
        class_name = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # 计算新的框的位置
        new_x = x
        new_y = y
        new_w = min(1, w + 2 * expansion_pixels / 3072)
        new_h = min(1, h + 2 * expansion_pixels / 2048)
        
        # 构建扩展后的行内容
        expanded_line = f"{class_name} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
        expanded_lines.append(expanded_line)
    
    # 将处理后的内容写回txt文件
    with open(txt_file_path, 'w') as f:
        f.writelines(expanded_lines)

# 遍历文件夹下所有txt文件并处理
def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(folder_path, filename)
            print(f"Processing {txt_file_path}...")
            expand_bounding_boxes(txt_file_path)
    print("Processing complete.")

# 要处理的文件夹路径
folder_path = '/home/tr/huawei_cloud_competition/data/test/renamed_txt'  # 替换为实际的文件夹路径
process_folder(folder_path)
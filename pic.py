# from PIL import Image
# import pillow_heif
# import os

# input_folder = "D:/pic"  # 你的 HEIC 文件夹路径
# output_folder = "D:/pic_output"  # 输出 PNG 文件夹路径

# os.makedirs(output_folder, exist_ok=True)

# for file in os.listdir(input_folder):
#     if file.lower().endswith(".heic"):     # 兼容 HEIC/Heic/HEIC
#         heic_path = os.path.join(input_folder, file)

#         # 生成新的 PNG 文件名
#         filename_no_ext = os.path.splitext(file)[0]
#         png_path = os.path.join(output_folder, filename_no_ext + ".png")

#         # 加载 HEIC
#         heif = pillow_heif.read_heif(heic_path)

#         # 转成 Pillow Image
#         img = Image.frombytes(
#             heif.mode,
#             heif.size,
#             heif.data,
#             "raw",
#         )

#         # **强制保存为 PNG**
#         img.save(png_path, format="PNG")

#         print(f"Converted: {file} -> {png_path}")

# print("全部转换完成！")



# import os
# from PIL import Image
# import pillow_heif

# input_folder = r"D:/pic"   # HEIC 文件夹
# output_folder = r"D:/pic-pdf"    # PDF 输出文件夹

# os.makedirs(output_folder, exist_ok=True)

# for file in os.listdir(input_folder):
#     if file.lower().endswith(".heic"):
#         heic_path = os.path.join(input_folder, file)

#         # 读取 HEIC
#         heif_file = pillow_heif.read_heif(heic_path)
#         img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")

#         # 保存为 PDF
#         pdf_name = os.path.splitext(file)[0] + ".pdf"
#         pdf_path = os.path.join(output_folder, pdf_name)
#         img.save(pdf_path, "PDF", resolution=100.0)

#         print(f"Converted: {file} -> {pdf_path}")

# print("全部 HEIC 已转换为 PDF！")



import os
from PyPDF2 import PdfMerger

def merge_pdfs_in_subfolders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 只处理文件夹
        if not os.path.isdir(folder_path):
            continue

        # 取子文件夹中的所有 PDF
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"❌ 子文件夹【{folder_name}】没有 PDF 文件，跳过")
            continue

        # 按文件名排序
        pdf_files.sort()

        merger = PdfMerger()
        print(f"🔧 正在合并：{folder_name}")

        for pdf in pdf_files:
            pdf_path = os.path.join(folder_path, pdf)
            merger.append(pdf_path)

        # 输出文件名：子文件夹名.pdf
        output_pdf = os.path.join(root_dir, f"{folder_name}.pdf")
        merger.write(output_pdf)
        merger.close()

        print(f"✅ 合并完成：{output_pdf}\n")


# ===== 使用示例 =====
root_directory = r"D:/pic-pdf"  # 修改成你的主目录路径
merge_pdfs_in_subfolders(root_directory)





# import os
# import re

# def remove_duplicate_pdfs(root_dir):
#     # 匹配结尾为 _数字.pdf，如 IMG_9308_2.pdf
#     duplicate_pattern = re.compile(r"^(.*)_(\d+)\.pdf$", re.IGNORECASE)

#     for folder_name in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue

#         print(f"📁 检查子文件夹：{folder_name}")

#         # 收集所有 PDF
#         pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

#         # 建立集合用于快速判断是否存在基础文件
#         pdf_set = set(pdf_files)

#         for filename in pdf_files:
#             match = duplicate_pattern.match(filename)
#             if match:
#                 base_name = match.group(1) + ".pdf"     # e.g. IMG_9308.pdf

#                 # 只有基础文件存在时才删除重复文件
#                 if base_name in pdf_set:
#                     file_path = os.path.join(folder_path, filename)
#                     print(f"🗑 删除重复文件：{filename}")
#                     os.remove(file_path)

#         print(f"✔ 完成 {folder_name} 的重复文件清理\n")


# # ===== 使用示例 =====
# root_directory = r"D:/pic-pdf"
# remove_duplicate_pdfs(root_directory)

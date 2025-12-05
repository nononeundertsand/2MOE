import subprocess
import sys
import os

# ==========================
# 配置你要执行的两个脚本
# ==========================
SCRIPT_1 = r"Qwen-7B/Qwen-7B_fine_tuning.py"
SCRIPT_2 = r"Mistral-7B/Mistral-7B_fine_tuning.py"


def run_script(script_path):
    """执行一个 Python 脚本，并实时输出其内容"""
    print(f"\n===== 正在执行：{script_path} =====\n")

    # 检查是否存在
    if not os.path.exists(script_path):
        print(f"错误：脚本不存在 -> {script_path}")
        sys.exit(1)

    # 执行脚本
    result = subprocess.run(
        [sys.executable, script_path],  # 使用当前 Python 解释器执行
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # 返回 exit code
    return result.returncode


if __name__ == "__main__":
    # 执行脚本 1
    code1 = run_script(SCRIPT_1)
    if code1 != 0:
        print(f"\n❌ {SCRIPT_1} 执行失败，中止后续任务。")
        sys.exit(1)

    print(f"\n✅ {SCRIPT_1} 执行成功，继续执行下一个脚本...\n")

    # 执行脚本 2
    code2 = run_script(SCRIPT_2)
    if code2 != 0:
        print(f"\n❌ {SCRIPT_2} 执行失败。")
        sys.exit(1)

    print("\n🎉 所有脚本执行完毕！")

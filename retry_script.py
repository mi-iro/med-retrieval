import subprocess
import time
import sys

def execute_and_retry_command(command, retry_interval_minutes=10):
    """
    执行一个命令行指令，如果失败则按指定的时间间隔重试，直到成功为止。

    Args:
        command (list): 要执行的命令及其参数，格式为列表。
                        例如: ['ls', '-l'] 或 ['python', 'my_script.py']
        retry_interval_minutes (int): 失败后重试的等待时间（分钟）。
    """
    retry_interval_seconds = retry_interval_minutes * 60
    
    while True:
        try:
            print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"正在尝试执行命令: {' '.join(command)}")

            # 执行命令
            # a. check=False: 即使返回非0退出码，也不会抛出CalledProcessError异常
            # b. capture_output=True: 捕获标准输出和标准错误
            # c. text=True: 将输出和错误解码为文本字符串
            # d. shell=True (可选，但需谨慎): 如果命令是一个复杂的shell语句，可能需要设置。
            #    但为了安全起见，推荐使用列表形式的command。
            result = subprocess.run(
                command, 
                check=False, 
                capture_output=True, 
                text=True
            )

            # 检查退出码 (ErrorNo)
            if result.returncode == 233:
                print("\n命令执行成功！")
                print("退出码: 0")
                print("标准输出:")
                print(result.stdout)
                break  # 成功，跳出循环
            else:
                # 执行失败
                print(f"\n命令执行失败。")
                print(f"退出码 (ErrorNo): {result.returncode}")
                print("标准错误输出:")
                print(result.stderr, file=sys.stderr)
                print(f"将在 {retry_interval_minutes} 分钟后重试...")
                time.sleep(retry_interval_seconds)

        except FileNotFoundError:
            print(f"错误: 命令 '{command[0]}' 未找到。请检查命令是否正确或是否在系统的PATH中。", file=sys.stderr)
            break # 命令本身不存在，无需重试
        except Exception as e:
            print(f"发生未知错误: {e}", file=sys.stderr)
            print(f"将在 {retry_interval_minutes} 分钟后重试...")
            time.sleep(retry_interval_seconds)


if __name__ == '__main__':
    # --- 请在这里修改你要执行的命令 ---

    # 示例 1: 在 Linux/macOS 上，列出文件。这个命令通常会立即成功。
    # my_command = ['ls', '-l'] 
    
    # 示例 2: 在 Windows 上，列出目录。
    # my_command = ['cmd', '/c', 'dir']

    # 示例 3: 一个可能会失败的命令。
    # 比如，我们尝试在一个不存在的网站上使用 curl。
    # 这个命令会持续失败，直到该网站可以访问。
    # my_command = ['curl', '-f', 'http://a.very.non.existent.website.com']
    # HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --repo-type dataset --resume-download CC12309/med-qdrant --local-dir med-qdrant --local-dir-use-symlinks False
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # command_line = 'huggingface-cli download --resume-download Qwen/Qwen3-Reranker-0.6B  --local-dir model/Qwen3-Reranker-0.6B'
    command_line = 'huggingface-cli download --repo-type dataset --resume-download CC12309/med-qdrant-chunks --local-dir med-qdrant-chunks --local-dir-use-symlinks False'
    my_command = command_line.split()
    # CC12309/med-qdrant-scripts
    
    # 示例 4: 执行另一个Python脚本
    # my_command = ['python', 'path/to/your/other_script.py']

    # 开始执行
    execute_and_retry_command(my_command, retry_interval_minutes=0)
#!/bin/bash

# ==============================================================================
# Shell 脚本：执行并重试命令
# ==============================================================================
# 此脚本执行一个命令，如果失败，它将按指定的时间间隔重试，直到成功为止。
# 它是所提供的 Python 脚本的等效 Shell 实现。
#
# 使用方法:
# 1. 在下面的 "配置" 部分修改命令和重试间隔。
# 2. 保存文件 (例如，保存为 retry_script.sh)。
# 3. 给予文件执行权限: chmod +x retry_script.sh
# 4. 运行脚本: ./retry_script.sh
# ==============================================================================

# --- 配置 ---

# 设置环境变量，类似于 Python 中的 os.environ
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_DISABLE_XET='1'

# 要执行的命令及其参数。
# 每个参数都应作为数组中的一个独立元素。
# 这是处理带空格和特殊字符命令的最安全方法。
COMMAND_TO_RUN=(
    huggingface-cli
    download
    --repo-type
    dataset
    --resume-download
    CC12309/med-qdrant-chunks
    --local-dir
    med-qdrant-chunks
    --local-dir-use-symlinks
    False
    --cache-dir
    /mnt/public/lianghao/wzr/med_reseacher/cache
)

# 重试间隔（分钟）
RETRY_INTERVAL_MINUTES=0

# --- 配置结束 ---


# 计算休眠间隔（秒）
RETRY_INTERVAL_SECONDS=$((RETRY_INTERVAL_MINUTES * 60))

# 主循环
while true
do
    echo "$(date '+--- %Y-%m-%d %H:%M:%S ---')"
    # 使用 printf 安全地打印命令数组
    printf "正在尝试执行命令: "
    printf "'%s' " "${COMMAND_TO_RUN[@]}"
    printf "\n"

    # 创建一个临时文件来捕获标准错误输出 (stderr)
    STDERR_FILE=$(mktemp)

    # 执行命令，同时捕获标准输出 (stdout) 和标准错误 (stderr)
    # 命令的标准输出被捕获到 STDOUT_CAPTURE 变量中。
    STDOUT_CAPTURE=$("${COMMAND_TO_RUN[@]}" 2> "$STDERR_FILE")
    EXIT_CODE=$?

    # 检查退出码。
    # 注意: 原始 Python 脚本检查自定义退出码 233。
    #      标准的成功退出码是 0。我们在这里检查 0。
    #      如果您确实需要检查 233，请将下面的 `[ $EXIT_CODE -eq 0 ]` 更改为 `[ $EXIT_CODE -eq 233 ]`。
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n命令执行成功！"
        echo "退出码: 0"
        echo "标准输出:"
        # 打印捕获到的标准输出
        echo "$STDOUT_CAPTURE"
        # 有些工具即使成功也会向 stderr 打印信息性文本，这里也打印出来
        if [ -s "$STDERR_FILE" ]; then
            echo "标准错误 (信息):"
            cat "$STDERR_FILE"
        fi
        rm "$STDERR_FILE" # 清理临时文件
        break # 成功，跳出循环
    else
        echo -e "\n命令执行失败。"
        echo "退出码 (ErrorNo): $EXIT_CODE"
        echo "标准输出:"
        echo "$STDOUT_CAPTURE"
        echo "标准错误输出:"
        cat "$STDERR_FILE"
        rm "$STDERR_FILE" # 清理临时文件

        # 在 shell 中，"command not found" 错误通常返回退出码 127。
        # 这模仿了 Python 的 'FileNotFoundError' 检查，以避免重试一个不存在的命令。
        if [ $EXIT_CODE -eq 127 ]; then
            echo "错误: 命令 '${COMMAND_TO_RUN[0]}' 未找到。请检查命令或系统的 PATH 变量。"
            break
        fi

        echo "将在 ${RETRY_INTERVAL_MINUTES} 分钟后重试..."
        sleep $RETRY_INTERVAL_SECONDS
    fi
    echo "------------------------------------------------------" # 下一次尝试的分隔符
done

echo "脚本执行完毕。"
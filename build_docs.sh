#!/bin/bash

# 定义你的 mkdocs 项目在WSL中的路径
MKDOCS_DOCS_PATH="/mnt/d/notes-webpage/My-Knowledge-Hub/docs/research/machine_learning"


# 转换命令保持不变，只是路径变量更新了
jupyter nbconvert --to markdown notebooks/*.ipynb --output-dir $MKDOCS_DOCS_PATH

echo "Markdown files have been generated in $MKDOCS_DOCS_PATH"
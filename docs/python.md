
## 创建虚拟环境
建议使用虚拟环境来隔离项目的依赖。你可以使用 venv、virtualenv 或者 conda 等工具来创建虚拟环境。
```commandline
python -m venv venv
```

### 激活虚拟环境：
要激活和取消激活 `venv` 虚拟环境，你需要使用命令行。下面是如何执行这些操作的步骤：
在 Windows 上：
```bash
venv\Scripts\activate
```
 在 macOS 或者 Linux 上：
```bash
source venv/bin/activate
```

### 取消激活虚拟环境：
执行这个命令将会退出虚拟环境，回到全局 Python 环境。
```bash
deactivate
```

执行这个命令将会退出虚拟环境，回到全局 Python 环境。
# Install Dependencies
```commandline
pip install langchain

```

OR Just Run Below Command
```commandline
pip install -r requirements.txt
```

要查看项目依赖的版本，可以使用以下命令：

```bash
pip freeze
```

这将列出当前激活的 Python 环境中所有安装的包及其版本信息。你也可以将输出重定向到一个文件中，例如：

```bash
pip freeze > requirements.txt
```

这样可以将当前环境的依赖版本信息保存到 `requirements.txt` 文件中，方便分享和重建环境。


# MART

## 1. 安装依赖
```bash
pip install -r requirements.txt
```
## 2. 参数设置
在 `config.py` 中设置以下关键参数：

- `retrain`:  
  - `True`：从头训练模型  
  - `False`：跳过训练，直接进行推理
- `DisLoss` 和 `AssLoss`：启用论文中提出的两种改进损失函数（根据需要开启）
## 3. 运行程序：
   ```bash
   python MART.py
   ```

**注意事项**：由于文件大小限制，已将 `new_dma_train` 分割为6个文件，下载后需要把数据文件重新合并再进行训练。

```markdown
# MART

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
2. 模型训练
   ```
   在config.py文件中有关于模型是否训练以及组件相关是否使用的参数设置，主要部分是retrain(true表示训练模型，false则会直接开始推理)，DisLoss和AssLoss是论文中提及的两个改进
   ```
3. 运行程序：
   ```在参数配置完成后运行下面代码即可
   python MART.py
   ```

**注意事项**：由于文件大小限制，已将 `new_dma_train` 分割为6个文件，下载后需要把数据文件重新合并再进行训练。

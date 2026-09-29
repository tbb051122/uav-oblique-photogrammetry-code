第四章 4.4.2 / 4.4.3 消融实验说明

一、实验目的
4.4.2 验证 RGB 颜色特征的作用；4.4.3 验证多尺度局部特征融合模块的作用。

二、训练设置（与 4.4.1 主模型完全一致）
数据：第四章_多样本训练\data（25 训练块 / 8 验证块 / 8 测试块，30 m 分块，5 cm 体素）
点数：8192/帧，batch size 4，学习率 0.001，40 轮，GPU：RTX 4060 Laptop

三、实验组
1. full      改进模型：XYZRGB + 多尺度融合（第四章_多样本训练\models\best_model.pth）
2. no_color  仅 XYZ 坐标，去掉颜色特征（第四章_消融实验\no_color\best_model.pth）
3. no_ms     去掉多尺度局部特征融合模块（第四章_消融实验\no_multi_scale\best_model.pth）

四、评估方式
测试集上用同一组固定采样点（5 个随机种子）各评估一次后取平均，
三个模型使用完全相同的采样点，指标按 4.3.2 节的 OA / mAcc / mIoU 计算。

五、结果
指标              full      no_color   no_ms
总体准确率 OA     0.5421    0.3002     0.5629
平均类别精度 mAcc 0.5579    0.3055     0.5662
平均交并比 mIoU   0.3992    0.1887     0.4006

逐类 IoU（建筑物/道路/植被/裸地/其他）
full     0.5104 / 0.2992 / 0.4276 / 0.5370 / 0.2218
no_color 0.4015 / 0.0762 / 0.2172 / 0.1109 / 0.1379
no_ms    0.5325 / 0.4289 / 0.4435 / 0.4524 / 0.1456

验证集最优 mIoU：full 0.3922；no_color 0.2279；no_ms 0.3781

六、结论
1. 颜色特征的作用非常明显：去掉颜色后 OA 从 0.5421 降到 0.3002，mIoU 从 0.3992
   降到 0.1887，道路 IoU 从 0.2992 降到 0.0762。
2. 多尺度融合模块：验证集上最优 mIoU 更高（0.3922 对 0.3781），裸地、其他人工设施
   两类 IoU 明显提高；但测试集上两组总体指标接近（mIoU 0.3992 对 0.4006，差异小于
   重复评估的标准差 0.004～0.008），本实验规模下没有显著差异。

七、文件
ablation_metrics.json              5 个种子逐次评估结果与均值/标准差
no_color\best_model.pth            无颜色特征模型权重
no_color\train_log.txt            训练日志
no_multi_scale\best_model.pth      无多尺度模块模型权重
no_multi_scale\train_log.txt       训练日志
fig4-11_颜色特征消融.png            论文图4-11
fig4-12_多尺度消融.png              论文图4-12

代码改动（Chapter4_RandLA_Net）：
model.py  RandLANet 增加 use_multi_scale 开关（默认 True，不影响原有权重）
train.py  train() 增加 use_multi_scale 参数
main.py   增加 --no_multi_scale 参数
data_utils.py 数据集增加 seed 参数（固定采样点，便于消融对比，默认 None 行为不变）
test.py   load_model()/test() 增加 use_multi_scale 参数

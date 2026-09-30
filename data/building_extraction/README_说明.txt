第四章 4.2.7 建筑目标提取结果说明

一、提取方法
先取语义类别为 1 的建筑点，再用 DBSCAN 按平面距离聚类，
最后统计每个建筑簇的二维范围和高度。

二、参数
building_class = 1
eps = 2.0 m
min_samples = 20
聚类前体素 = 0.2 m

三、Tile_74（论文表 4-12 采用该瓦片，0.2 m 体素下采样、eps=2.0 m、min_samples=20）
模型预测标签：提取 15 个目标（≥20 m² 的 6 个），占地总面积 2824.66 m²，
              最大目标面积 1241.85 m²、最大高度 42.93 m；模型预测的建筑点占比 18.3%。
参考标签：    39 个目标（≥20 m² 的 14 个），占地总面积 4221.35 m²，
              最大目标面积 1545.61 m²、最大高度 44.84 m；建筑点占比 27.46%
              （见 data/preclassification/Tile_74_preclassify_report.txt）。
模型提取的目标数明显少于参考标签（15 对 39，少约 62%；占地总面积少约 33%），
主要原因是语义分割对同类建筑点存在漏检，以及相邻建筑在聚类时发生粘连，
小面积目标（<20 m²）更容易被漏掉。

本目录文件：
Tile_74_building_targets.csv / Tile_74_summary.json   模型侧目标清单与汇总
Tile_74_reference_summary.json                        参考标签侧汇总
Tile_74_iou_pairing.json                              模型与参考目标配对包围盒 IoU（0.4～0.9）

四、Tile_37（早期小规模验证，非论文定稿数值）
结果一：基于模型预测标签（本目录 Tile_37_building_targets.csv）
输入：
C:\Users\tongb\Desktop\佟邦本\毕业设计\第四章_训练测试结果\predictions\Tile_37_predicted_full.npy
输出：
C:\Users\tongb\Desktop\佟邦本\毕业设计\第四章_建筑提取结果\Tile_37_全瓦片\building_targets.csv
提取到 11 个目标。
其中 cluster 2 面积 542.19 平方米、cluster 9 面积 133.94 平方米、
cluster 8 面积 62.66 平方米，属于比较可信的建筑目标；
cluster 1 高度 37.57 米但面积只有 8.19 平方米，可能是杆状物或误判，
需要结合 CloudCompare 目视检查。

结果二：基于参考标签（本目录 Tile_37_reference_targets.csv）
输入：
C:\Users\tongb\Desktop\佟邦本\毕业设计\第四章_预分类结果\Tile_37_annotation_005m_preclassified.las
输出：
C:\Users\tongb\Desktop\佟邦本\毕业设计\第四章_建筑提取结果\Tile_37_参考标签\building_targets.csv
提取到 18 个目标。
其中 cluster 2 有 8685 点、面积 896.28 平方米、高度 39.82 米，
cluster 9 有 7262 点、面积 440.88 平方米，属于主要建筑；
其余小簇可能是附属结构、建筑物碎片或聚类参数保留下来的噪声。

五、建议
1. 用 CloudCompare 打开提取结果对应的点云，重点检查高而细的簇；
2. eps 和 min_samples 可以按论文描述用 k 近邻距离分布进一步标定，
   再比较不同参数下的建筑目标数量与边界完整性。

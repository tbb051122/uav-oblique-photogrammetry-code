点云预分类结果说明
数据来源：
E:\毕设\毕设\Productions\Production_1\Tile_37.las

1. Tile_37_annotation_005m.las
   从 Tile_37.las 按 5 cm 体素下采样得到的标注样本，共 3,257,310 个点，
   保留原始 RGB 和 EPSG:4545 坐标，分类字段初始为 0。
2. Tile_37_annotation_005m_preclassified.las
   预分类结果，保留原始 RGB，Classification 字段为 0-5。
3. Tile_37_annotation_005m_class_colors.las
   按类别着色的检查版，建议先用 CloudCompare 打开这一份目视检查。
4. Tile_37_annotation_005m_preclassify_report.txt
   分类统计、特征分位数和预分类参数。

类别编号：
0 未分类
1 建筑物
2 道路
3 植被
4 裸地
5 其他人工设施

类别颜色：
1 建筑物      红色
2 道路        深灰
3 植被        绿色
4 裸地        棕色
5 其他人工设施 黄色

预分类脚本：
C:\Users\tongb\Desktop\佟邦本\毕业设计\Chapter4_RandLA_Net\preclassify_heuristic.py

其他 LAS 运行示例：
python Chapter4_RandLA_Net\preclassify_heuristic.py --input 输入文件.las --out-dir 输出目录


生产耗时与磁盘性能实测说明（对应论文表 4-4、表 4-5）

一、文件清单

  aerial_triangulation_timing.csv   空中三角测量耗时（剔除前 52 min、剔除后 29 min）
  las_production_timing.csv         LAS 点云生产耗时（15 h 55 min / 14 h 7 min）
  osgb_production_timing.csv        OSGB 模型生产耗时（1 h 28 min / 1 h 44 min）
  disk_benchmark.csv                实验所用两块磁盘的实测读写性能

二、耗时的测定方法

  1. 用时以 ContextCapture（iTwin Capture Modeler）生产项目界面上显示的“处理时间”为准，
     并与成果目录中文件的时间范围核对一致。
  2. 开始、结束时间取成果目录内所有文件的最早、最晚修改时间（含 Data 下各 Tile_* 子目录）。
  3. 平均写入速率 = 成果体积 ÷ 用时；该值反映生产全过程的平均落盘速率，
     包含瓦片计算、纹理打包等计算时间，因此明显低于磁盘的顺序写入上限。
  4. 生产成果目录：
       剔除前  E:\biyesheji\qian1\Productions\Production_2          111 个切块
       剔除后  I:\biyesheji\剔除冗余影像后\Productions\Production_2   74 个切块
     LAS 成果目录：
       剔除前  I:\biyesheji\qian1\Productions\Production_1          111 个瓦片
       剔除后  I:\biyesheji\hou1\Productions\Production_1           74 个瓦片

三、磁盘实测方法

  1. 顺序读写：单个 1 GiB 文件，1 MiB 块，Windows 无缓存标志
     （FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH）逐块读写，取全过程平均速率。
  2. 小文件写：400 个 256 KiB 文件逐个写入并 fsync 落盘，统计总吞吐与文件数速率。
  3. 结果（MB/s，1 MB = 10^6 字节）：
       E盘（内置 KIOXIA EXCERIA PLUS G3，NVMe）顺序写 2407、顺序读 2693、小文件写 284
       I盘（外接 Lenovo F309 Lite，USB）       顺序写 31.3、顺序读 39.0、小文件写 2.8
  4. 外接盘吞吐接近 USB 2.0 链路上限，若改用 USB 3.0 线缆/接口重新测试，
     顺序读写应能提高到数百 MB/s，届时表 4-5 的数值需要同步更新。

四、与论文结论的关系

  剔除后切块数由 111 减少到 74（-33.3%），但 OSGB 生产耗时由 1 h 28 min 变为 1 h 44 min。
  原因是两次导出写入的磁盘不同（剔除前为内置 NVMe，剔除后为外接 USB 盘），
  且剔除后一次导出时读取重建成果与写入瓦片数据共用同一条 USB 链路；
  两次导出的平均落盘速率仅 0.87 MB/s 与 0.66 MB/s，说明该阶段耗时主要取决于
  模型规模与逐瓦片处理，输出磁盘性能差异进一步削弱了影像数量减少带来的收益。

  注：LAS 点云生产的两次运行均在 I 盘（外接盘）上完成，因此该阶段的对比不受磁盘差异影响。

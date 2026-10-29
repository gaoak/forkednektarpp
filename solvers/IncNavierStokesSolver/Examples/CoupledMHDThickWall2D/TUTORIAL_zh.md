# Hunt 构型准静态 MHD：学生使用教程

主要网格路线仍是 **Mesh2D 的 square 分支**。几何网格未改变：二维 x/y 方管截面和上下厚导电壁。流速改为 z 向 w，磁场改为 y 向 By；IncNS 使用准三维 Homogeneous1D 零模态，ADR 求二维截面电势。

上下是有限电导率 Hartmann 壁，左右是绝缘侧壁；所有外电边界绝缘，无外加电极电压。本例是厚壁、有限电导率 Hunt 构型，壁电导率并非无穷大。

## 1. 模型、参数和进程分工

流体区域是 0<=x,y<=1；下固体 -0.25<=y<=0，上固体 1<=y<=1.25。网格仍为 4 个流体、4 个固体四边形。所有 x/y 流体边界都是无滑移壁面，没有 x 向入口和出口。

```text
 y=1.25   +--------------------------------+  固体外表面绝缘
          |          上导电厚壁            |
 y=1      +--------------------------------+  上 Hartmann 流固界面
          |          流体截面              |  w 沿 z，垂直纸面
          |    左右绝缘侧壁；By 沿 y       |
 y=0      +--------------------------------+  下 Hartmann 流固界面
          |          下导电厚壁            |
 y=-0.25  +--------------------------------+  固体外表面绝缘
          x=0                            x=1
```

| 参数 | 默认值与含义 |
| --- | --- |
| Homogeneous / HomModesZ / LZ | 1D / 2 / 1，z 周期方向，仅充分发展零模态 |
| 变量 | IncNS: u,v,w,phi,p；ADR: phi，接收三分量速度 |
| 速度和电势展开 | 每方向 5 模态、7 求积点；压力 4 模态、同样求积点 |
| Ha / HalfWidth | 1 / 0.5，以流体半高为特征长度 |
| Kinvis / ElectricConductivity | 0.1 / 1 |
| WallConductivity / WallThickness | 10 / 0.25 |
| Flowrate | 截面平均 w=0.1，控制器沿 z 驱动 |
| TimeStep / NumSteps | 0.02 / 500，计算至 t=10 |

归一化密度为 1，定义 Ha=B*a*sqrt(sigma/Kinvis)，因此 By=Ha/a*sqrt(Kinvis/sigma)≈0.632455532。不要直接把 By=1 当成 Ha=1。壁电导比 sigma_wall*h/(sigma_fluid*a)=5。

参考设置来自 `/StorageLBM/yujiez/Shercliff_Ha1000/N5/Duct.xml`。与原 Shercliff 的四面绝缘不同，本例显式求解上下导电固体。直接系数传输要求速度和流体电势使用相同展开，因此这里 phi 与速度均为 5 模态，压力仍可为 4 模态。

IncNS 在准三维空间求 u,v,w，ADR 因 d(phi)/dz=0 只求二维截面电势。电流 J=sigma*(u×B-grad(phi))，三分量速度必需，因为 (u×B)_x=-w*By。ADR 返回的 phi 系数放入流体的实零 Fourier 模态。

HomModesZ=2 在这里表示实零模态和空虚零模态，不是两个独立轴向网格单元。当前耦合只支持此充分发展状态，不支持非零 Fourier 模态的扰动；保持默认 npz=1，MPI 只划分 x/y 单元。

流体 M 块，固体另分 N 块；IncNS 用 M 进程，ADR 用 M+N 进程，总数为 2M+N。默认 M=2,N=1，共 5 进程。配对流体块交换场系数，固体仍参加 ADR 的全局电势求解通信。

## 2. 准备编译和运行环境

### 2.1 进入源码根目录并加载模块

本机路径如下，换机器时替换第一行。使用 Bash；每个新终端都要重新加载模块。

```bash
cd /home/agao/testcode/nektar
module load nektar++/latest
```

模块加载后 PATH 中可能是另一个安装版本的求解器。下面通过当前 build 的完整路径运行程序。

### 2.2 编译

本机已有配置好的 `./build`，通常直接运行：

```bash
cmake --build build --target MHDPartition ADRSolver IncNavierStokesSolver -j 4
cmake --build build --target FieldConvert -j 4
```

新构建目录需要先配置：

```bash
cmake -S . -B build \
    -DNEKTAR_USE_MPI=ON -DNEKTAR_USE_CWIPI=ON \
    -DNEKTAR_USE_SCOTCH=ON -DNEKTAR_UTILITY_EXTRAS=ON \
    -DNEKTAR_UTILITY_FIELDCONVERT=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
```

CWIPI 可使用已有兼容安装，配置 `CWIPI_LIBRARY`、`CWIPI_INCLUDE_DIR`、`THIRDPARTY_BUILD_CWIPI=OFF`；也可以使用项目的 ThirdParty 构建。新机器上的具体依赖路径由教师确认。不要覆盖、删除现有 build 的全部配置来安装依赖。CWIPI、MPI、Nektar++ 必须使用相互兼容的 MPI 环境。

### 2.3 设置本教程的路径变量

在源码根目录执行：

```bash
source solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh
```

此脚本只设置变量，不生成网格、不运行求解。后文命令在同一个已加载环境的终端中执行。

| 变量 | 含义 |
| --- | --- |
| `$mhd_repo_dir`、`$mhd_case_dir` | 源码根目录、本算例目录 |
| `$mhd_build_dir` | 构建目录，默认源码下的 build |
| `$mhd_partitioner` | 当前 build 的 MHDPartition |
| `$mhd_fluid_solver`、`$mhd_electric_solver` | 当前 build 的 IncNS、ADR |
| `$mhd_fieldconvert`、`$mhd_nekmesh` | 当前 build 的后处理、网格转换工具 |

Debug 构建的可执行文件通常有 `-g` 后缀，脚本会自动处理。检查文件是否存在：

```bash
ls -l "$mhd_partitioner" "$mhd_fluid_solver" "$mhd_electric_solver"
```

使用其他构建目录时，先设置 `export NEKTAR_BUILD_DIR=/绝对路径/build`，再 source `env.sh`。

## 3. 先运行默认 Hunt 算例

在完成第 2 节环境设置后执行：

```bash
"$mhd_case_dir/Allrun" "$mhd_build_dir/student-hunt-ha1"
```

脚本自动复制输入、统一预分区、MPI 启动、导出截面 CSV 并进行物理一致性检查。需要已编译 FieldConvert。

```text
PASS Hunt Ha=1: 501 electrical solves; mean w=0.0999999877, max w=0.206953; Lorentz/Joule balance=3.001e-07
```

末位数值会随环境变化。501 次电势求解包括初始化和 500 步积分。此处没有提供 Hunt 精确解析场，日志中的 L2 error 对默认零参考场计算，是场范数，不是离散误差，不能再套用旧通道的 1e-8 误差要求。

M=1,N=1 的较少进程运行：

```bash
FLUID_RANKS=1 SOLID_RANKS=1 \
    "$mhd_case_dir/Allrun" "$mhd_build_dir/student-hunt-m1-n1"
```

改分区数时换新的输出目录。短暂瞬态演示可以减少 NumSteps，但不要期待通过默认稳态检查；用 MHD_CHECK_RESULTS=0 跳过专用检查，并另行判断结果。

## 4. 用 Mesh2D square 方法画完整流固网格

### 4.1 先认识 square 分支

Mesh2D 本机位置为 `/home/agao/testcode/Mesh2D`。可只读检查：

```bash
git -C /home/agao/testcode/Mesh2D branch --list square
git -C /home/agao/testcode/Mesh2D show square:mesh.cpp
git -C /home/agao/testcode/Mesh2D show square:edgefunctions.h
```

其基本流程是：`edgefunctions.h` 定义四条参数边和分段数 → `RectRegion::MeshGen` 生成结构四边形 → `MeshRegions` 定义边界 → 输出 XML。

当前 square 的坐标、加密分布用于其他方形算例，不直接等同于本教程的坐标。这里提供两个小参考文件，沿用相同接口：

- [scripts/mesh2d_edgefunctions.h](scripts/mesh2d_edgefunctions.h)：本算例的坐标与边函数。
- [scripts/mesh2d_mesh.cpp](scripts/mesh2d_mesh.cpp)：生成一张完整流固矩形网格，输出 `full.xml`。

### 4.2 一条命令准备教学网格

```bash
bash "$mhd_case_dir/scripts/build_mesh2d_case.sh" \
    /home/agao/testcode/Mesh2D "$mhd_build_dir/student-mesh2d"
```

脚本读取 `square` 分支记录的 CAD2D 子模块版本，导出到自己的工作目录，复制本教程参考 C++ 和头文件，在该目录编译、生成完整网格，再提取匹配的两个网格并放入会话文件。**它不会切换或修改原 Mesh2D 工作区的分支。**

要求本机 Mesh2D 仓库有 `square` 分支，CAD2D 子模块仓库中已有该分支引用的提交。缺少子模块时请在自己的 Mesh2D 副本中完成仓库和子模块准备；本脚本不会自动联网下载。

目录大致为：

```text
student-mesh2d/
├── CAD2D/                    square 所用库的独立副本
├── mesh.cpp                  教学参考主程序
├── edgefunctions.h           学生可修改的坐标和边函数副本
├── mesh-g                    网格生成程序
├── full.xml                  完整流固网格
├── log.mesh2d                网格生成和质量检查日志
└── inputs/
    ├── fluid-mesh.xml
    ├── electric-mesh.xml
    ├── fluid.xml
    └── electric.xml
```

本机验证默认输出为 15 个点、22 条边、8 个四边形，提取后流体 4 个、固体 4 个。`log.mesh2d` 中应有 0 个孤立点、0 个负 Jacobian。12 条 isolated edges 是矩形外边界的单元边数量，是正常的，不要求为零。

### 4.3 坐标数组怎样对应网格

参考头文件核心内容：

```cpp
const std::array<double, 3> x_nodes = {{0.0, 0.5, 1.0}};
const std::array<double, 5> y_nodes = {{-0.25, 0.0, 0.5, 1.0, 1.25}};
const int Nx = static_cast<int>(x_nodes.size()) - 1;
const int Ny = static_cast<int>(y_nodes.size()) - 1;
```

数组保存的是**几何单元边界坐标**。x 有 3 个节点，形成 2 列；y 有 5 个节点，形成 4 行；总数 Nx×Ny=8。它们不是单元内部的高阶求积点。

上下两行是固体，中间两行是流体。**y=0 和 y=1 必须恰好是整行网格节点，单元不能跨越材料界面。**

`RectRegion` 在 s∈[-1,1] 上均匀采样。`coordinateFromNodes` 把每个参数区间映射到给定的相邻坐标，所以可以通过非均匀坐标数组进行局部加密。坐标须严格递增。

四条边函数沿用 square 的名称和方向：

| 函数 | 位置 | 参数方向 |
| --- | --- | --- |
| `edge03(s)` | 完整矩形底边 | 左→右 |
| `edge39(s)` | 右边 | 下→上 |
| `edge69(s)` | 顶边 | 左→右 |
| `edge06(s)` | 左边 | 下→上 |

主程序传给 `RectRegion` 的顺序是底、右、顶、左。保留相同接口和方向，可以避免误把顶边再倒序造成连通性问题。

### 4.4 主程序和 XML 输出

参考主程序的主要调用是：

```cpp
RectRegion region(edges, "full_fluid_solid");
region.MeshGen(Nx, Ny);
MeshRegions combined("domain", 1e-10);
combined.AddRegion(region);
combined.defineBoundary(reinterpret_cast<void *>(edge03), Nx, 1);
combined.defineBoundary(reinterpret_cast<void *>(edge69), Nx, 2);
combined.defineBoundary(reinterpret_cast<void *>(edge06), Ny, 3);
combined.defineBoundary(reinterpret_cast<void *>(edge39), Ny, 4);
combined.CheckMesh();
combined.outXml("full.xml");
combined.outCOMPO("full.xml", {0});
```

`outXml` 后必须调用 `outCOMPO`，后者补齐 composite、domain 和 XML 结束标签。本教程编译时不加 `OUTPUTEXP`，展开由独立求解器会话设置。

注意 `defineBoundary` 的逻辑边界 ID 与最终 composite ID 可能不同：当前 CAD2D 会顺序输出 composites。不要假设传入 1 就必然得到 C[1]。下一步按单元位置和几何编号提取，避免依赖这些来源分组号。

### 4.5 为什么只画一次完整网格

直接系数传输要求两套网格的流体单元保持相同的顶点、边、单元 ID 和方向。只保证坐标相同还不够。

参考流程为：生成一张完整流固 `full.xml`，然后从中保留原编号提取流体网格。不要分别调用两次 Mesh2D 生成流体与流固网格，也不要在提取后重新紧凑编号。

提取脚本是 [scripts/prepare_mesh.py](scripts/prepare_mesh.py)，也可以手动调用：

```bash
python3 "$mhd_case_dir/scripts/prepare_mesh.py" \
    "$mhd_build_dir/student-mesh2d/full.xml" \
    "$mhd_build_dir/student-mesh2d/inputs"
```

默认把完全落在 `0<=y<=1` 的单元归为流体，完全落在外侧的单元归为固体；跨越界面的单元会报错。外边界和流体上下壁按坐标识别，输出规范的 composites。

脚本支持普通未压缩、未分区 XML 的二维直边四边形，针对上下壁矩形拓扑；其他几何应扩展判定规则。可以用 `--fluid-ymin`、`--fluid-ymax` 改材料界面位置，但还要同步修改会话和边界、解析解。

### 4.6 启动 Mesh2D 生成的算例

```bash
MHD_CASE_DIR="$mhd_build_dir/student-mesh2d/inputs" \
FLUID_RANKS=2 SOLID_RANKS=1 \
    "$mhd_case_dir/Allrun" "$mhd_build_dir/student-mesh2d/run"
```

应该通过 Hunt 流量、场对应关系和能量平衡检查。输入目录和输出目录分开，建议都写绝对路径。

### 4.7 修改和加密网格

练习时修改工作副本 `student-mesh2d/edgefunctions.h`，例如把 x 改为 `{0,0.25,0.5,0.75,1}` 并将 `std::array` 长度改为 5。保留 y=0、1，即可加密而不改变物理几何。

修改后在工作目录重编译并重新提取：

```bash
cd "$mhd_build_dir/student-mesh2d"
c++ -std=c++11 -O2 -I . mesh.cpp \
    CAD2D/CompositEdge.cpp CAD2D/LineEdge.cpp CAD2D/MeshRegion.cpp \
    CAD2D/MeshRegions.cpp CAD2D/RectRegion.cpp CAD2D/tinyxml2.cpp \
    CAD2D/util.cpp -o mesh-g
./mesh-g > log.mesh2d 2>&1
python3 "$mhd_case_dir/scripts/prepare_mesh.py" full.xml inputs
```

然后用新输出目录运行 `Allrun`。不要重新执行准备脚本覆盖自己的工作副本；`build_mesh2d_case.sh` 会重新复制默认参考文件。也可以先复制整个教学目录作为新的练习目录。


## 5. 输入文件怎样配置

四个输入仍为 fluid-mesh.xml、electric-mesh.xml、fluid.xml、electric.xml。前两者存几何与 composites，后两者存方程、变量、展开、边界、材料、参数与耦合。网格未改，仅会话改成 Hunt 流动。

### 5.1 区域与边界

| Composite | 流体网格 | 电网格 |
| --- | --- | --- |
| C[0] | 流体体单元 | 流体体单元 |
| C[1] | 不使用 | 上下导电固体体单元 |
| C[10] | y=0，下无滑移壁 | y=-0.25，下固体外表面绝缘 |
| C[11] | y=1，上无滑移壁 | y=1.25，上固体外表面绝缘 |
| C[12] | x=0，左无滑移侧壁 | x=0，全域绝缘侧边 |
| C[13] | x=1，右无滑移侧壁 | x=1，全域绝缘侧边 |
| DOMAIN | C[0] | C[0,1] |

这是提取后的规范 composite 编号，不一定等于 Mesh2D 原始 full.xml 的编号。两边材料界面共享点、边和单元方向；ADR 不在内部流固界面施加外部边界条件。

### 5.2 IncNS 的准三维设置

```xml
<I PROPERTY="EQType" VALUE="UnsteadyNavierStokes" />
<I PROPERTY="SolverType" VALUE="VelocityCorrectionScheme" />
<I PROPERTY="Projection" VALUE="Continuous" />
<I PROPERTY="Homogeneous" VALUE="1D" />
<I PROPERTY="ElectricPotential" VALUE="Coupled" />
```

参数中有 HomModesZ=2、LZ=1、Flowrate=0.1。变量严格按 u,v,w,phi,p 排列，phi 位于 p 前。

所有四条 x/y 壁都给 u=v=w=0；压力使用 `<N VAR="p" USERDEFINEDTYPE="H" VALUE="0" />`。不再给 x 方向抛物线入口，也没有出口压力边界。周期 z 向的平均流量由控制器调整等效驱动压力梯度，输出到 partitions/fluid.prs。

```xml
<EXPANSIONS>
    <E COMPOSITE="C[0]" NUMMODES="5" NUMPOINTS="7"
       FIELDS="u,v,w,phi" TYPE="MODIFIED" />
    <E COMPOSITE="C[0]" NUMMODES="4" NUMPOINTS="7"
       FIELDS="p" TYPE="MODIFIED" />
</EXPANSIONS>
```

FORCING 中保留 ForcingLorentz 并设 IMPLICITDAMPING=True。外加场函数为 `<E VAR="By" VALUE="MagneticFieldStrength" />`。默认无 Ex/Ey/Ez，去掉旧通道中维持抛物线解析解的 BodyForce。

### 5.3 ADR 电势与材料

ADR 保持二维，没有 Homogeneous=1D，因为充分发展状态 phi 不随 z 变化。它只声明求解变量 phi，展开为 5 模态、7 求积点，与流体侧 phi 相同。

```xml
<ELECTRICPOTENTIAL>
    <FLUID COMPOSITE="C[0]" CONDUCTIVITY="ElectricConductivity" />
    <SOLID COMPOSITE="C[1]" CONDUCTIVITY="WallConductivity" />
</ELECTRICPOTENTIAL>
```

EQType=CoupledElectricPotential，Projection=Continuous。各单元恰好属于一个材料，电导率为正常数。所有外电边界设 `<N VAR="phi" VALUE="0" />`，表示总法向电流为零，不是电极电压。固体电势不为零，内部电流通过厚壁闭合。

纯 Neumann 电势只确定到加性常数，代码使用已有奇异系统处理固定参考值。检查梯度、电流或电势差，避免把不同参考值造成的常数偏差当作物理差异。

### 5.4 耦合设置

两边用 Coefficients 类型，IncNS 的 Role=Fluid，ADR 的 Role=Electric，FluidRanks=M。流体发送列表为 u,v,w,mhd_time,mhd_stop，接收 phi；ADR 反过来。

mhd_time、mhd_stop 是协议数据，不加入 VARIABLES。ADR 的二维网格需要接收 w，因为电势右端含 -w*By。预分区工具会生成含正确 Role、FluidRanks 的完整分区会话。

速度和电势直接交换单元局部展开系数；实零模态的布局只在初始化时验证一次。固体仍参加全局电势求解，运动源项只在流体单元更新。

改源 XML 后必须重新预分区，旧分区目录不会自动更新。

## 6. 不用 Allrun，手动预分区和启动

先加载 module 和 source scripts/env.sh。在单独工作目录放入四个输入，例如：

```bash
mkdir -p "$mhd_build_dir/student-hunt-manual"
cp "$mhd_case_dir/fluid-mesh.xml" "$mhd_case_dir/fluid.xml" \
   "$mhd_case_dir/electric-mesh.xml" "$mhd_case_dir/electric.xml" \
   "$mhd_build_dir/student-hunt-manual/"
cd "$mhd_build_dir/student-hunt-manual"
```

使用单个进程预处理：

```bash
"$mhd_partitioner" 2 1 partitions \
    fluid-mesh.xml fluid.xml electric-mesh.xml electric.xml > log.partition 2>&1
cat partitions/partition-map.csv
```

流体分区目录应有 2 个 XML，电分区目录有 3 个。流体单元两边 rank 一一对应，固体 incns_rank=-1。启动时在同一个 MPI 作业中运行两个程序：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mpiexec -np 2 "$mhd_fluid_solver" --cwipi fluid "$PWD/partitions/fluid_xml" \
    : -np 3 "$mhd_electric_solver" --cwipi electric "$PWD/partitions/electric_xml" \
    > log.coupled-mhd 2>&1
```

这是一条命令，反斜杠用于换行，可以写成一行。冒号分隔两个应用；不要在两个独立终端各启动一次 mpiexec。两个 --cwipi 名称须不同。

M=2、N=2 时预处理用 2 2，启动分别用 -np 2、-np 4。保持默认 npz=1，不在均匀方向拆 MPI 进程。直接输入两个分区目录，不再额外附加原始会话。

后处理并检查：

```bash
"$mhd_fieldconvert" -f -m homplane:planeid=0 \
    fluid-mesh.xml fluid.xml partitions/fluid.fld fluid.csv
"$mhd_fieldconvert" -f electric-mesh.xml electric.xml \
    partitions/electric.fld electric.csv
python3 "$mhd_case_dir/check_results.py" log.coupled-mhd
```

检查脚本用于本默认矩形、纯绝缘外边界、充分发展 Hunt 场景。改成有外部电功输入、移动壁、不同拓扑或很短瞬态时，需相应修改验证方法。

## 7. 结果位置和可视化

直接运行 Allrun 的默认输出目录仍为 `build/coupled-mhd-thick-wall-2d`。传入路径时输出到指定目录。

| 文件 | 内容 |
| --- | --- |
| log.partition / log.coupled-mhd | 预分区与求解日志 |
| partitions/partition-map.csv | 单元对应的材料、进程 |
| partitions/fluid_xml、electric_xml | 两套预分区完整会话 |
| partitions/fluid.fld/ | 准三维 u,v,w,phi,p |
| partitions/electric.fld/ | 二维 phi,Jx,Jy,Jz,sigma |
| partitions/fluid.prs | 流量控制器的驱动压力梯度 |
| fluid.csv / electric.csv | 专用检查器使用的逐单元截面采样 |
| log.convert-fluid / log.convert-electric | 自动导出日志 |

并行 fld 是含 Info.xml 和每进程数据的目录，需完整保留。

在工作目录导出截面 VTU：

```bash
"$mhd_fieldconvert" -f -m homplane:planeid=0 \
    fluid-mesh.xml fluid.xml partitions/fluid.fld fluid-mean.vtu
"$mhd_fieldconvert" -f electric-mesh.xml electric.xml \
    partitions/electric.fld electric.vtu
```

在 ParaView 打开后查看 w，不是 u。sigma 在流体为 1、固体为 10。phi 在界面连续，法向斜率可以跳变。CSV 和截面图使用 homplane 提取，避免把未提取的准三维场直接当成二维点场导出。

默认 mean w≈0.1，采样 max w≈0.206953，后期驱动梯度≈0.309465712516。Ha=1、8 单元的教学结果用于检查耦合和物理一致性，不代表高 Ha 情形或网格收敛后的精确 Hunt 曲线。

默认检查验证：u/v 接近零、四壁无滑移、流量正确、两求解器流体 phi 对应、导电壁存在电流、后期驱动梯度稳定，以及

```text
积分_流体 w*(J×B)_z + 积分_流体和固体 |J|^2/sigma = 0。
```

洛伦兹功率应为负，焦耳耗散为正。CSV 有限输出精度下，默认相对平衡阈值是 1e-4。它不是精确速度剖面误差指标。

## 8. Allrun 参数和队列运行

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| FLUID_RANKS / SOLID_RANKS | 2 / 1 | M、N，总进程 2M+N |
| MHD_CASE_DIR | 算例目录 | 四个输入 XML 的目录 |
| MHD_CHECK_RESULTS | 1 | 默认 Hunt 物理检查；0 跳过 |
| MHD_TIMEOUT | 120 秒 | 求解超时，长计算需增加 |
| MHD_TRANSFER | Coefficients | 本 Hunt 准三维例仅支持该模式 |
| FIELD_CONVERT | 当前 build 中工具 | 覆盖后处理程序路径 |
| NEKTAR_BUILD_DIR | 源码下 build | 指定构建目录 |

自定义输入与输出分开，例如：

```bash
MHD_CASE_DIR=/绝对路径/my-inputs MHD_TIMEOUT=3600 \
FLUID_RANKS=2 SOLID_RANKS=2 \
    "$mhd_case_dir/Allrun" /绝对路径/my-hunt-m2-n2
```

Slurm 示例：保存到源码根目录 student-hunt.sbatch，再从根目录提交 sbatch。队列、账户、MPI 集成方式按本机配置调整；模板未在队列提交验证。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=coupled-hunt
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-hunt-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load nektar++/latest
source solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh
export FLUID_RANKS=2
export SOLID_RANKS=1
"$mhd_case_dir/Allrun" "$mhd_build_dir/student-hunt-${SLURM_JOB_ID}"
```

Allrun 内部已有 mpiexec，不再嵌套 MPI 启动。修改 M、N 时同步修改 ntasks。某些集群需要专用 srun 多程序配置，应遵照本机规定。

## 9. 修改算例和常见问题

- 网格加密：改 Mesh2D 坐标数组及其长度，保留材料界面节点；重新生成、提取、预分区。
- p 加密：两边流体 phi、速度的阶数和求积同步，压力可保持低一阶；重新预分区。
- 提高 Ha：磁场按 Ha 参数计算。高 Ha 的 Hartmann 层约为半高/Ha，不能用当前粗网格直接算 Ha=1000。
- 改壁电导率：同步会话参数并检查电流；有限厚壁不等于无限导电边界。
- 改几何：同步 HalfWidth、材料判定和边界位置。默认检查器针对矩形、直边四边形，其他几何需扩展。
- 改成 z 依赖扰动：当前只支持 HomModesZ=2 零模态，不支持改成 4/8 等一般 Fourier 模态。需要另行扩展耦合。

| 现象 | 检查 |
| --- | --- |
| 未注册 CoupledElectricPotential/Coefficients | 是否调用当前 build 的程序，CWIPI 是否开启 |
| Paired fluid layouts differ | 单元编号、方向、模态、求积是否一致，是否忘了重新预分区 |
| HomModesZ=2 zero mode 提示 | 是否改变均匀展开模式，是否使用 npz 拆分 |
| 电势 L2 error 不小 | 本例未提供精确场，该行是对零参考的范数；看专用物理检查 |
| 运行等待 | 查看日志，确认同一个 mpiexec、正确 np 和分区目录 |
| 后处理失败 | FieldConvert 是否已编译、是否先用 homplane 提取截面 |
| 很短运行后检查失败 | 默认检查要求达到稳态，增加步数或用 MHD_CHECK_RESULTS=0 并自行验证 |
| fld 是目录 | 并行输出正常，保留完整目录 |

## 10. 参考脚本索引与回归测试

| 文件 | 用途 |
| --- | --- |
| [Allrun](Allrun) | 完整启动和检查流程 |
| [scripts/env.sh](scripts/env.sh) | source 后获得当前 build 工具路径 |
| [scripts/build_mesh2d_case.sh](scripts/build_mesh2d_case.sh) | 用 square 对应 CAD2D 版本生成独立教学网格 |
| [scripts/mesh2d_mesh.cpp](scripts/mesh2d_mesh.cpp) | 结构矩形网格主程序 |
| [scripts/mesh2d_edgefunctions.h](scripts/mesh2d_edgefunctions.h) | 坐标数组与边函数 |
| [scripts/prepare_mesh.py](scripts/prepare_mesh.py) | 保留编号提取流体和完整电网格 |
| [generate_case.py](generate_case.py) | 生成当前 Hunt 会话和相同的固定 8 单元网格 |
| [check_results.py](check_results.py) | 流量、电势对应和能量等物理检查 |
| [../../Tests/test_coupled_hunt.py](../../Tests/test_coupled_hunt.py) | 不同 MPI 分区一致性、拒绝不支持的 Fourier 配置 |

旧的电极驱动二维通道精确解保存在 `../../Tests/CoupledMHDChannel/`，用于保留原有点传输和系数回归。它的物理方向、外电边界和判据与当前 Hunt 不同，不要混用会话。

```bash
cmake --build build --target MHDCoefficientTransferCheck Tester -j4
ctest --test-dir build \
    -R '^IncNavierStokesSolver_(CoupledHunt|MHDCoefficients|CoupledMHDThickWall2D)$' \
    --output-on-failure
```

修改源码后先重新配置 CMake 以注册新测试。学生日常计算不需要逐次执行全部回归。

建议练习：对比 M/N=1/1、2/1、2/2 的 w；加密网格并比较平均流量和驱动梯度；绘制 w、phi、Jx/Jy 和 sigma，解释壁内闭合电流；比较有限壁电导与绝缘壁；明确区分物理一致性验证与精确剖面、网格收敛验证。

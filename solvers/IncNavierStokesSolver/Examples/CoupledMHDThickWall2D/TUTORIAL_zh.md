# 厚导电壁面准静态 MHD：学生使用教程

本教程使用 **Mesh2D 的 `square` 分支**的建网格方法，完成生成完整流固网格、提取匹配网格、设置输入、统一预分区、MPI 启动以及结果检查。建议先跑通第 3 节现成算例，再按第 4 节自己生成网格。

适用源码版本：具有 `CoupledElectricPotential`、`MHDPartition` 和 `Coefficients` 耦合类型的版本。环境模块提供依赖，实际运行当前源码对应的 `build` 中的程序。

## 1. 问题和程序分工

### 1.1 两个求解器计算什么

| 程序 | 计算区域 | 求解/保存的变量 | 接收 | 发送 |
| --- | --- | --- | --- | --- |
| `IncNavierStokesSolver`，简称 IncNS | 流体 | 速度 u、v，压力 p，以及接收的电势 phi | 流体区域的电势系数 | 当前时间积分阶段的速度系数 |
| `ADRSolver`，选择 `CoupledElectricPotential` | 流体和固体 | 电势 phi | 流体速度系数 | 流体区域的电势系数 |

电流、电势满足

\[
\boldsymbol J=\sigma(\boldsymbol E_0+\boldsymbol u\times\boldsymbol B-\nabla\phi),
\qquad \nabla\cdot\boldsymbol J=0.
\]

固体静止，取 u=0。无外加体电场、材料内部电导率恒定时，固体内部是拉普拉斯方程。流固界面共享几何实体，ADR 的连续 Galerkin 求解同时处理电势连续和法向电流守恒。

本教程是沿 x 流动、壁面法向为 y、磁场沿 z 的二维平面通道。它没有表示方管全部三维侧壁；完整三维方管需要另建三维网格。

### 1.2 默认几何和参数

```text
                 x=0                x=0.5                x=1
 y=1.25          +--------------------+--------------------+ 上电极 phi=0.1
                 |       固体        |       固体         |
 y=1             +--------------------+--------------------+ 上流固界面
                 |       流体        |       流体         |
 y=0.5           +--------------------+--------------------+
                 |       流体        |       流体         |
 y=0             +--------------------+--------------------+ 下流固界面
                 |       固体        |       固体         |
 y=-0.25         +--------------------+--------------------+ 下电极 phi=0
```

| 项目 | 默认值 |
| --- | --- |
| 流体区域 | `0 <= x <= 1, 0 <= y <= 1` |
| 固体厚度 | 上、下壁各 0.25 |
| 网格 | 流体 4 个、固体 4 个四边形 |
| 单元内展开 | 每方向 `NUMMODES=4, NUMPOINTS=5, TYPE=MODIFIED` |
| `Kinvis` | 0.1 |
| `ElectricConductivity`、`WallConductivity` | 1、0.1 |
| `U0`、`B0`、`Voltage` | 均为 0.1 |
| `TimeStep`、`NumSteps` | 0.001、5 |

这是归一化验证算例。`NUMMODES=4` 表示每方向 4 个模态，通常对应最高三次多项式，不代表几何网格只有 4 个点。网格加密与提高单元内展开阶数是两件事。

### 1.3 M、N 和进程数

流体分成 M 块，两个程序复用同一组流体分区；固体另分 N 块，仅 ADR 使用。

| M | N | IncNS 进程 | ADR 进程 | 总 MPI 进程 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 2 | 3 |
| 2 | 1 | 2 | 3 | 5，默认配置 |
| 2 | 2 | 2 | 4 | 6 |

总进程数是 **2M+N**。M、N 均须大于零，每块至少一个相应材料的单元。

IncNS 本地进程 i 与 ADR 本地进程 i 拥有相同流体单元，直接传输展开系数。ADR 剩下 N 个纯固体进程不接收速度场，不更新运动电动势源项，但仍参与 ADR 内部控制同步、界面通信和全局电势求解。

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

## 3. 先运行现成的 8 单元算例

不用画网格，直接运行：

```bash
"$mhd_case_dir/Allrun" "$mhd_build_dir/student-first-run"
```

`Allrun` 复制四个输入文件，生成统一预分区，在同一个 MPI 作业里启动两个求解器，然后比较解析解。默认 M=2、N=1，共 5 个 MPI 进程。

成功时末尾类似：

```text
PASS: 6 electrical solves; maximum analytical L2 error 9.089e-13
```

末位误差可因环境而不同，检查门槛是五条变量 L2 误差均小于 `1e-8`。5 个流体时间步有 6 次电势求解是正常的，初始化也需要电势；更一般的交换次数还取决于积分阶段数。

用更少进程：

```bash
FLUID_RANKS=1 SOLID_RANKS=1 \
    "$mhd_case_dir/Allrun" "$mhd_build_dir/student-m1-n1"
```

改变 M、N 时用新输出目录，防止残留旧分区。

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

应该通过相同解析解检查。输入目录和输出目录分开，建议都写绝对路径。

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

## 5. 输入文件和区域定义

四个输入文件分工如下：

| 文件 | 内容 |
| --- | --- |
| `fluid-mesh.xml` | 仅流体几何与区域 |
| `electric-mesh.xml` | 流体和固体完整几何与区域 |
| `fluid.xml` | 流体方程、变量、展开、边界、参数和耦合 |
| `electric.xml` | 电势方程、材料、展开、电边界和耦合 |

`GEOMETRY` 中 `VERTEX` 定义坐标，`EDGE` 引用顶点，`ELEMENT/Q` 引用四条边，`COMPOSITE` 把实体归组，`DOMAIN` 指定体单元区域。

### 5.1 提取后的 composite 编号

| Composite | 流体网格 | 电网格 |
| --- | --- | --- |
| C[0] | 全部流体体单元 | 全部流体体单元 |
| C[1] | 不使用 | 全部固体体单元 |
| C[10] | 下流体壁 y=0 | 下外电极 y=-0.25 |
| C[11] | 上流体壁 y=1 | 上外电极 y=1.25 |
| C[12] | 流体左入口 | 全域左边界 |
| C[13] | 流体右出口 | 全域右边界 |
| DOMAIN | C[0] | C[0,1] |

这是提取脚本输出的规范编号，不能用来解释原始 Mesh2D `full.xml` 的 composite 编号。两会话里的 C[10] 可以对应不同位置；配对要求针对的是流体单元与展开布局。

### 5.2 IncNS 会话

展开定义：

```xml
<EXPANSIONS>
    <E COMPOSITE="C[0]" NUMMODES="4" NUMPOINTS="5"
       FIELDS="u,v,phi,p" TYPE="MODIFIED" />
</EXPANSIONS>
```

`CONDITIONS/SOLVERINFO` 中：

```xml
<I PROPERTY="EQType" VALUE="UnsteadyNavierStokes" />
<I PROPERTY="SolverType" VALUE="VelocityCorrectionScheme" />
<I PROPERTY="Projection" VALUE="Continuous" />
<I PROPERTY="ElectricPotential" VALUE="Coupled" />
```

变量顺序为 `u,v,phi,p`，电势放在压力前；完整三维是 `u,v,w,phi,p`。直接系数模式要求两边流体速度、电势的基函数、模态数、求积点和单元布局相同。

保留 `FORCING` 中的 `ForcingLorentz` 和引用的 `ElectricMagneticFields`。默认上下流体壁无滑移；入口和出口都给抛物线速度 `u=4*U0*y*(1-y),v=0`，出口压力为零，其余按原 XML 设置高阶压力边界。这是验证用的稳态边界组合，实际问题的出口条件应重新选择。

流体 XML 中对 phi 的边界不会再强加到 ADR 返回的电势上，但会话仍应包含合法的变量边界配置。

### 5.3 ADR 会话

ADR 只声明求解变量 phi，接收的速度属于辅助数组：

```xml
<EXPANSIONS>
    <E COMPOSITE="C[0,1]" NUMMODES="4" NUMPOINTS="5"
       FIELDS="phi" TYPE="MODIFIED" />
</EXPANSIONS>
<ELECTRICPOTENTIAL>
    <FLUID COMPOSITE="C[0]" CONDUCTIVITY="ElectricConductivity" />
    <SOLID COMPOSITE="C[1]" CONDUCTIVITY="WallConductivity" />
</ELECTRICPOTENTIAL>
```

`SOLVERINFO` 中为 `EQType=CoupledElectricPotential`、`Projection=Continuous`。每个体单元恰好属于一个材料，电导率是正的常数表达式。可以增加多个 SOLID 条目，但不在流固内部界面施加外部边界条件。

默认下电极 phi=0，上电极 phi=Voltage；左右外边界为 `<N VAR="phi" VALUE="0" />`，代表**总法向电流为零**。非零 Neumann 值对应外向电流密度的负值，不能直接当成电势法向导数。

### 5.4 参数与外加场

两份会话的相关物性和外加场应一致。默认函数只有 `<E VAR="Bz" VALUE="B0" />`。加固定电场时可在两份会话对应函数中增加 Ex/Ey/Ez，同时调整物理边界、初始场和解析解。

XML 的 PARAMETERS 按顺序求值，先定义基础参数，再定义引用它的 Resistance、JyExact 等。把新参数写在引用处后面会导致 `Unknown parameter`。

ADR 将固定电场弱载荷在初始化时缓存，每次仅在流体单元更新运动源项；电边界仍在请求时刻重新求值。固体省去动态源项不表示其电势恒为零。

### 5.5 耦合块由预处理生成

源会话保留已提供的 `TYPE="Cwipi"` 模板。MHDPartition 会生成新的完整分区会话，改为 `TYPE="Coefficients"`，加入 Role、FluidRanks，去掉插值专用选项。

生成后的流体块类似：

```xml
<COUPLING NAME="mhd" TYPE="Coefficients">
    <I PROPERTY="Role" VALUE="Fluid" />
    <I PROPERTY="FluidRanks" VALUE="2" />
    <I PROPERTY="SendSteps" VALUE="1" />
    <I PROPERTY="ReceiveSteps" VALUE="1" />
    <I PROPERTY="SendVariables" VALUE="u,v,mhd_time,mhd_stop" />
    <I PROPERTY="ReceiveVariables" VALUE="phi" />
</COUPLING>
```

ADR 的 Role 为 Electric，收发列表反过来。mhd_time、mhd_stop 是内部协议数据，不加入求解变量；三维速度列表在 v 后加 w。系数模式的控制信息放在消息元数据中。

输入文件改动后必须重新预分区，旧分区不会自动跟随源文件更新。

## 6. 手动调用：看清预分区和启动过程

这一节展示 Allrun 实际做的步骤。先完成第 2 节环境设置，然后从已经准备好的 Mesh2D 输入开始：

```bash
mkdir -p "$mhd_build_dir/student-manual"
cp "$mhd_build_dir/student-mesh2d/inputs/"*.xml "$mhd_build_dir/student-manual/"
cd "$mhd_build_dir/student-manual"
```

### 6.1 一次生成两套统一分区

```bash
"$mhd_partitioner" 2 1 partitions \
    fluid-mesh.xml fluid.xml electric-mesh.xml electric.xml \
    > log.partition 2>&1
```

参数依次是 M、N、输出目录、流体网格、流体会话、电网格、电会话。预处理用单进程运行，不需要在外面加 `mpiexec -np 5`。

```bash
cat log.partition
cat partitions/partition-map.csv
ls partitions/fluid_xml
ls partitions/electric_xml
```

流体目录应有 2 个分区 XML，电目录应有 3 个。CSV 中，流体单元的 incns_rank 等于 adr_rank；固体的 incns_rank=-1，adr_rank 从 M 开始。Scotch 分区组合可因设置而不同，关键是流体配对一致。

### 6.2 在同一个 MPI 作业里同时启动

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mpiexec \
    -np 2 "$mhd_fluid_solver" --cwipi fluid "$PWD/partitions/fluid_xml" \
    : -np 3 "$mhd_electric_solver" --cwipi electric "$PWD/partitions/electric_xml" \
    > log.coupled-mhd 2>&1
```

冒号是 MPI 多程序启动语法，前面程序用 2 个进程，后面用 3 个进程，属于同一个作业。不要在两个终端分别运行两次 mpiexec。

`--cwipi fluid` 和 `--cwipi electric` 为两应用建立不同的本地通信器；即使使用直接系数传输，也保留这种启动机制。输入是两个**分区目录**，不再附加原始会话文件，也不让两个程序各自重新划分网格。

M=2、N=2 时，预分区参数改为 `2 2`，启动进程数分别改为 `-np 2` 和 `-np 4`，总数是 6。

### 6.3 检查日志

另开终端可用 `tail -f /绝对路径/student-manual/log.coupled-mhd` 查看进度；计算完成后运行：

```bash
python3 "$mhd_case_dir/check_results.py" log.coupled-mhd
```

正常日志包含两次 `layout validated once`，表示两个应用完成启动布局检查。L2 误差应有 IncNS 的 u、v、phi、p 和 ADR 的 phi 共五条；末尾打印电势求解次数。检查脚本只适用于配套解析算例，不是任意物理问题的通用正确性判据。

## 7. 输出文件、解析解和可视化

### 7.1 输出结构

```text
run/
├── fluid-mesh.xml, fluid.xml             输入副本
├── electric-mesh.xml, electric.xml       输入副本
├── log.partition
├── log.coupled-mhd
└── partitions/
    ├── partition-map.csv
    ├── fluid_xml/                       M 份流体会话
    ├── electric_xml/                    M+N 份电会话
    ├── fluid.fld/                       并行流体场
    └── electric.fld/                    并行电场
```

并行 `.fld` 通常是目录，包含 Info.xml 和各进程数据。保留整个目录，不能只复制 P0000000.fld。流体输出有 u、v、phi、p；电输出有 phi、Jx、Jy、Jz、sigma。

### 7.2 导出 VTU

在加载模块、source env.sh 后执行：

```bash
cd "$mhd_build_dir/student-mesh2d/run"
"$mhd_fieldconvert" fluid-mesh.xml fluid.xml partitions/fluid.fld fluid.vtu
"$mhd_fieldconvert" electric-mesh.xml electric.xml partitions/electric.fld electric.vtu
```

这两条命令已用本教程生成的 Mesh2D 网格验证。使用原始未分区的完整几何与会话读取并行场目录；不要用单独一个分区 XML 代表全域。

在 ParaView 打开 fluid.vtu 或 electric.vtu，点击 Apply 后选择变量。先检查 sigma：流体应为 1，固体为 0.1；再看速度、电势和电流。电势在流固界面连续，电势斜率可以跳变。电流按单元局部输出，某些分量允许有材料界面跳变，不要依靠自动平滑后的图像判断守恒。

### 7.3 默认解析解

流体速度为 `u=4*U0*y*(1-y),v=0`，最大 u 为 0.1；压力参考解 p=0。设壁厚 h，则

\[
R=2h/\sigma_s+1/\sigma_f,\qquad
J_y=-\frac{\mathrm{Voltage}+2U_0B_0/3}{R}
\simeq-0.0177777777778.
\]

令 s=clamp(y,0,1)，整个流固区域的电势是

\[
\phi=-J_y\left[(y+h)/\sigma_s+(1/\sigma_f-1/\sigma_s)s\right]
      -U_0B_0(2s^2-4s^3/3).
\]

该解在每个材料区域内是低阶多项式，小网格也可精确表示，因此主要检查实现和通信。真实非多项式问题仍需要网格与阶数收敛研究。

## 8. Allrun 参数和队列启动

### 8.1 常用环境变量

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| FLUID_RANKS、SOLID_RANKS | 系数模式 2、1 | M、N |
| MHD_CASE_DIR | 本算例目录 | 四个输入 XML 所在目录 |
| MHD_CHECK_RESULTS | 1 | 1 检查配套解析解，0 跳过 |
| MHD_TIMEOUT | 120 秒 | 求解超时，大算例需增大 |
| MHD_TRANSFER | Coefficients | 也可用 Cwipi 做旧点传输对比 |
| NEKTAR_BUILD_DIR | 源码下 build | 构建目录，建议用绝对路径 |
| INCNS_SOLVER、ADR_SOLVER、MHD_PARTITIONER | 自动选择 build 中程序 | 手动覆盖各程序路径 |

第一个位置参数是输出目录。输入和输出目录分开；改变进程数用新输出目录。

运行已正确设置好的非解析问题，例如：

```bash
MHD_CASE_DIR=/绝对路径/my-inputs \
MHD_CHECK_RESULTS=0 MHD_TIMEOUT=3600 \
FLUID_RANKS=4 SOLID_RANKS=2 \
    "$mhd_case_dir/Allrun" /绝对路径/my-run-m4-n2
```

此例需要 10 个 MPI 进程，并要求每个材料区域有足够单元。关闭解析检查不表示验证已完成，应自行检查收敛、守恒及网格和时间步独立性。

### 8.2 Slurm 模板

以下模板可保存到源码根目录 `student-mhd.sbatch`。队列、账户和 MPI 启动集成方式依集群配置调整；模板未在实际队列提交验证。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=coupled-mhd
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-mhd-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load nektar++/latest
source solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh
export FLUID_RANKS=2
export SOLID_RANKS=1
"$mhd_case_dir/Allrun" "$mhd_build_dir/student-slurm-${SLURM_JOB_ID}"
```

在源码根目录执行 `sbatch student-mhd.sbatch`。Allrun 已含 mpiexec，不在外面再包一层 MPI。改 M、N 后同步修改 ntasks。有些集群需要专用 srun 多程序配置，应按本机要求替换启动方式，不能直接照搬 mpiexec 冒号语法。

## 9. 修改算例时保持哪些一致

1. **改分区数**：重新预分区，换输出目录，核对总进程数 2M+N。
2. **只加密网格**：改 Mesh2D 坐标数组及长度，保留界面节点，重新生成、提取、预分区。几何尺寸不变可继续检查原解析解。
3. **改展开阶数**：两份会话中流体的 NUMMODES、NUMPOINTS、基函数同步修改，重新预分区。
4. **改物性、磁场、固定电场**：同步改相关会话参数，并核对初始场、体力、电边界和解析解是否仍匹配。
5. **改几何**：同步改坐标、材料判定和边界识别、厚度参数及解析函数。默认解析式按流体高度 1 编写，不能把高度改成 2 后照用。
6. **换实际问题**：替换验证用边界和初始场，删除或替换不适用的 ExactSolution 和 BodyForce，采用自己的验证标准。

默认 BodyForce 中的 `8*Kinvis*U0-JyExact*B0` 用来维持解析抛物线稳态，不是所有 MHD 问题通用的体力。

实现目前针对静态完整二维/三维网格、静止固体、恒定外加电磁场。实际教学验证是二维直边四边形；移动电网格、均匀 Fourier 扩展不能直接套用。

## 10. 常见问题

| 现象 | 检查方向 |
| --- | --- |
| 找不到 MHDPartition 目标 | 是否启用 Scotch、NEKTAR_UTILITY_EXTRAS 并编译 |
| 方程或 Coefficients 未注册 | 是否运行旧安装程序，当前 build 是否启用 CWIPI |
| 库找不到或版本不一致 | 是否在新终端重新加载模块，程序与库是否同一构建 |
| square/CAD2D 提交找不到 | Mesh2D 仓库和其子模块是否完整，square 分支是否存在 |
| 提取时提示单元跨界面 | y 数组是否含 y=0、1，是否使用了未对齐材料的单元 |
| 提取后边界 composite 为空 | 是否仍为上下壁矩形拓扑，左右边界是否竖直，界面位置是否正确 |
| Paired fluid layouts differ | 流体单元 ID、方向、基函数、阶数、求积点是否一致，是否忘了重新预分区 |
| 分区为空/文件数不匹配 | M/N 是否超过对应单元数，是否使用了旧分区目录 |
| 运行等待或超时 | 看重定向日志，两个程序是否同一个 mpiexec，np 是否匹配分区目录 |
| Unknown parameter | 先定义基础参数再引用，检查拼写和大小写 |
| 修改输入似乎无效 | MHD_CASE_DIR 是否指向自己的输入，求解器是否还读旧分区 |
| 解析检查失败 | 五条误差是否完整；解析解、体力、边界是否仍配套，不只提高容差 |
| fld 是目录 | 并行输出正常形式；保留完整目录，用 FieldConvert 读取 |

## 11. 参考脚本索引

| 脚本 | 用途与用法 |
| --- | --- |
| [Allrun](Allrun) | 复制输入、预分区、MPI 启动、解析检查；第 3、8 节 |
| [scripts/env.sh](scripts/env.sh) | 用 source 设置当前 build 的工具路径；第 2.3 节 |
| [scripts/build_mesh2d_case.sh](scripts/build_mesh2d_case.sh) | 从 square 对应 CAD2D 版本准备独立教学网格；第 4.2 节 |
| [scripts/mesh2d_mesh.cpp](scripts/mesh2d_mesh.cpp) | RectRegion→MeshRegions→完整 XML 的参考主程序 |
| [scripts/mesh2d_edgefunctions.h](scripts/mesh2d_edgefunctions.h) | 节点数组、四边参数函数参考 |
| [scripts/prepare_mesh.py](scripts/prepare_mesh.py) | 保留几何编号提取匹配网格；第 4.5 节 |
| [generate_case.py](generate_case.py) | 无外部网格依赖的固定 8 单元 Python 生成器 |
| [check_results.py](check_results.py) | 对给定求解日志检查本算例解析误差与求解次数 |
| [../../Tests/test_mhd_coefficients.py](../../Tests/test_mhd_coefficients.py) | 开发回归，检查分区、系数传输、非零固定电场及拒绝错误布局 |

`generate_case.py` 将四个文件写到脚本所在目录并覆盖同名输入。练习时先复制到自己的 inputs 再运行。其拓扑和单元选择是固定的，不能只修改数组长度而不修改选择逻辑。

测试目标已启用时，在源码根目录可执行：

```bash
cmake --build build --target MHDCoefficientTransferCheck Tester -j 4
ctest --test-dir build \
    -R '^IncNavierStokesSolver_(MHDCoefficients|CoupledMHDThickWall2D)$' \
    --output-on-failure
```

不需要学生每次运行都重做全部回归测试。

## 12. 可选参考：Gmsh 和旧点传输模式

主要教学路线使用 Mesh2D。额外保留 [thick_wall_channel.geo](scripts/thick_wall_channel.geo)，供需要 Gmsh 的同学参考。其 Physical Group 按模板定义，不能与 Mesh2D 原始 composite 编号混用。使用时需编译 NekMesh，再执行：

```bash
cmake --build "$mhd_build_dir" --target NekMesh -j 4
mkdir -p "$mhd_build_dir/student-gmsh/inputs"
gmsh -2 "$mhd_case_dir/scripts/thick_wall_channel.geo" \
    -format msh2 -o "$mhd_build_dir/student-gmsh/channel.msh"
"$mhd_nekmesh" "$mhd_build_dir/student-gmsh/channel.msh" \
    "$mhd_build_dir/student-gmsh/full.xml:xml:uncompress"
python3 "$mhd_case_dir/scripts/prepare_mesh.py" \
    "$mhd_build_dir/student-gmsh/full.xml" "$mhd_build_dir/student-gmsh/inputs" \
    --source-format gmsh
cp "$mhd_case_dir/fluid.xml" "$mhd_case_dir/electric.xml" \
    "$mhd_build_dir/student-gmsh/inputs/"
```

接着按第 4.6 节改 MHD_CASE_DIR 运行。Gmsh 结构化四边形与命令行原理可查 [官方教程](https://gmsh.info/doc/texinfo/gmsh.html#t6)，其线上的 Transfinite 数量是节点数，不是单元数。Mesh2D 路线不需要 Gmsh 或 NekMesh 转换。

旧点传输模式可用于比较：

```bash
MHD_TRANSFER=Cwipi FLUID_RANKS=2 ELECTRIC_RANKS=2 \
    "$mhd_case_dir/Allrun" "$mhd_build_dir/student-point-transfer"
```

此模式独立设置两程序进程数，不用 SOLID_RANKS；输出为 fluid-mesh.fld、electric-mesh.fld，路径与系数模式不同。

## 13. 建议学生提交的练习材料

1. 保存 Mesh2D 输入副本、网格检查日志、partition-map.csv、求解日志。
2. 比较 M/N=1/1、2/1、2/2，解释总进程数和流体块的配对关系。
3. 保持物理参数、几何尺寸不变，加密坐标数组，比较 h 加密与增加 NUMMODES。
4. 提交 sigma、phi、Jy 图，解释电势连续、斜率跳变和电流方向。
5. 画出 IncNS↔ADR 的配对流体数据通信，以及 ADR 内部流固耦合求解通信。

# CoupledMHD 零基础快速上手

> 面向第一次接触 Nektar++ 和 MHD 的同学。
> 目标：**15 分钟内跑通算例**，知道每一步在做什么，能看懂结果、改参数，并在报错时知道先查哪里。
> 更完整的原理、网格生成和输入文件说明见 [TUTORIAL_zh.md](TUTORIAL_zh.md)。

---

## 0. 一分钟版本

如果你已经在本文档第 2 节准备好的机器上，只要执行：

```bash
export REPO=/storage/debian2home3/MHD/yujiez/forkednektarpp
export BUILD=/home/yujiez/build-forkednektarpp
export NEKTAR_BUILD_DIR="$BUILD"

source "$REPO/solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh"
"$mhd_case_dir/Allrun" "$BUILD/student-first-run"
```

看到最后一行类似下面这样，就成功了：

```text
PASS: 6 electrical solves; maximum analytical L2 error 9.089e-13
```

下面的内容解释这条命令背后的每一步。

---

## 1. 先认识这个算例

### 1.1 它算什么

一条二维平面通道，流体在中间，上下各有一层**导电固体壁**，磁场沿 z 方向。流体运动切割磁感线产生电流，电流又反过来影响流动。程序要同时求出：

- 流体的速度 `u, v` 和压力 `p`；
- 整个流体 + 固体区域里的电势 `phi`；
- 由电势得到的电流密度 `Jx, Jy, Jz`。

### 1.2 为什么要两个求解器

| 程序 | 负责区域 | 求的变量 |
| --- | --- | --- |
| `IncNavierStokesSolver`（简称 IncNS） | 只有流体 | `u, v, p` |
| `ADRSolver`（`EQType=CoupledElectricPotential`） | 流体 + 固体 | `phi` |

两个程序在同一个 MPI 作业里同时运行，每一步交换数据：

```text
IncNS  ──── 发送 u, v ────►  ADR
IncNS  ◄─── 发送 phi  ────  ADR
```

### 1.3 两套网格和 M、N

- `fluid-mesh.xml`：只包含流体单元；
- `electric-mesh.xml`：包含流体 + 固体全部单元。

**关键要求：电网格里的流体部分，必须和流体网格使用完全相同的单元 ID、顶点 ID、连接关系和方向。** 最稳妥的做法是先画一张完整网格，再从中"原样保留编号"提取流体部分，不要分别画两次。

流体被分成 **M** 块，每个程序的进程各拿一块；固体另外分成 **N** 块，只有 ADR 使用：

| M | N | IncNS 进程 | ADR 进程 | 总进程 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 2 | 3 |
| 2 | 1 | 2 | 3 | **5（默认）** |
| 2 | 2 | 2 | 4 | 6 |

总进程数 = **2M + N**。M、N 都必须大于 0，且每块至少有一个单元。

### 1.4 预分区是什么

直接拿原始 XML 启动是不行的。要先运行一次 `MHDPartition`：

1. 把流体和固体单元分给各个进程；
2. 为每个进程写出一份"只含自己单元"的 XML；
3. 把输入文件里的点插值耦合 `TYPE="Cwipi"` 改写成直接系数耦合 `TYPE="Coefficients"`。

之后每个进程读取属于自己的那份 XML，进程之间直接传展开系数，不再做点插值。

---

## 2. 准备环境（二选一）

下面两条路线只需要走通一条。

### 路线 A：直接使用已经编译好的构建（当前 ubuntu22 构建机）

```bash
source /usr/local/tools/modules/init/bash
export MODULEPATH="/apps/modulefiles:$MODULEPATH"
module load cmake/3.24.2 boost/1.74.0 hdf5/1.12.3 scotch/7.0.1 \
            lapack/3.7.1 openmpi/4.1.0 zlib/1.2.9
```

确认程序存在：

```bash
ls -l /home/yujiez/build-forkednektarpp/solvers/IncNavierStokesSolver/IncNavierStokesSolver
ls -l /home/yujiez/build-forkednektarpp/solvers/ADRSolver/ADRSolver
ls -l /home/yujiez/build-forkednektarpp/utilities/Extras/MHDPartition
```

> 这是 **ubuntu22 这台机器本地**的构建目录。如果你登录的是另一台机器（例如 `debain-91`），这个目录不存在，请走路线 B。

### 路线 B：在自己的 Debian/Ubuntu 机器上编译（只需做一次）

安装系统依赖（Debian 12 / Ubuntu 22.04 均可）：

```bash
sudo apt update
sudo apt install -y \
    build-essential gfortran cmake git patch python3 \
    openmpi-bin libopenmpi-dev \
    libboost-iostreams-dev libboost-program-options-dev libboost-system-dev \
    liblapack-dev \
    libscotch-dev libptscotch-dev \
    libtinyxml-dev zlib1g-dev \
    libhdf5-mpi-dev
```

> `libhdf5-mpi-dev` 是可选但推荐的并行 HDF5。不装也能编：CMake 会自动下载并编译 ThirdParty HDF5，只是首次配置更慢。
> CWIPI 不用单独安装，CMake 会自动下载 `cwipi-0.11.1` 并编译。**首次配置需要联网**；如果下载中断，删掉 `build/ThirdParty/` 里对应的 0 字节文件后重新配置。

编译：

```bash
cd /path/to/forkednektarpp          # 换成你的源码目录
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEKTAR_USE_MPI=ON \
    -DNEKTAR_USE_CWIPI=ON \
    -DNEKTAR_USE_SCOTCH=ON \
    -DNEKTAR_UTILITY_EXTRAS=ON \
    -DNEKTAR_UTILITY_FIELDCONVERT=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
cmake --build build --target MHDPartition ADRSolver IncNavierStokesSolver FieldConvert -j "$(nproc)"
```

编译完成后，`build/` 下会有：

```text
build/solvers/IncNavierStokesSolver/IncNavierStokesSolver
build/solvers/ADRSolver/ADRSolver
build/utilities/Extras/MHDPartition
build/utilities/FieldConvert/FieldConvert
```

后面把 `BUILD` 指到这个 `build` 目录即可。

### 2.1 设置本教程要用的变量

每开一个新终端都要重新做一次（路线 A 还要先加载模块）：

```bash
export REPO=/storage/debian2home3/MHD/yujiez/forkednektarpp   # 源码目录
export BUILD=/home/yujiez/build-forkednektarpp                # 构建目录
export NEKTAR_BUILD_DIR="$BUILD"

source "$REPO/solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh"
```

> 在 `debain-91` 上，源码通常挂在 `/home3/MHD/yujiez/forkednektarpp`；构建目录用你自己 configure 出来的那个，例如 `REPO/build`。路径以实际机器为准。

`env.sh` 只设置变量，不运行任何计算。它提供的常用变量：

| 变量 | 含义 |
| --- | --- |
| `$mhd_case_dir` | 本算例目录（四个输入 XML 所在处） |
| `$mhd_build_dir` | 构建目录 |
| `$mhd_partitioner` | `MHDPartition` |
| `$mhd_fluid_solver` | `IncNavierStokesSolver` |
| `$mhd_electric_solver` | `ADRSolver` |
| `$mhd_fieldconvert` | `FieldConvert` |

> 换机器时只改 `REPO` 和 `BUILD` 两行。**不要**把 `BUILD` 指向一个旧的安装目录（例如 `build/dist`），否则可能运行到不含本功能的旧程序。

---

## 3. 第一次运行

```bash
"$mhd_case_dir/Allrun" "$BUILD/student-first-run"
```

`Allrun` 会依次做四件事：

1. 复制四个输入 XML 到输出目录；
2. 用默认的 M=2、N=1 生成统一预分区；
3. 用一个 `mpiexec` 同时启动 2 个 IncNS 进程和 3 个 ADR 进程；
4. 用 `check_results.py` 检查解析解误差。

成功时末尾是：

```text
PASS: 6 electrical solves; maximum analytical L2 error 9.089e-13
```

解释：

- **6 次电势求解**：5 个时间步，加上初始化时的一次；
- **误差 `9.089e-13`**：与解析解的 L2 误差，检查门槛是小于 `1e-8`；
- 具体数值因机器和并行划分略有差异，只要 PASS 即可。

再用更小规模跑一次（M=1、N=1，共 3 个进程）：

```bash
FLUID_RANKS=1 SOLID_RANKS=1 \
    "$mhd_case_dir/Allrun" "$BUILD/student-m1-n1"
```

> **换进程数时一定要换一个新的输出目录**，否则旧分区文件会和新设置冲突。

---

## 4. 手动走一遍（理解 Allrun）

建议第一次照着做一遍，之后出问题就知道卡在哪一步。

```bash
export RUN="$BUILD/student-manual"
mkdir -p "$RUN"
cp "$mhd_case_dir"/{fluid-mesh.xml,fluid.xml,electric-mesh.xml,electric.xml} "$RUN"/
cd "$RUN"
```

### 4.1 预分区

```bash
"$mhd_partitioner" 2 1 partitions \
    fluid-mesh.xml fluid.xml electric-mesh.xml electric.xml \
    > log.partition 2>&1

cat log.partition
```

应看到：

```text
Unified Scotch partition: 2 fluid + 1 solid partitions; 5 total MPI ranks.
```

参数含义依次是：**M、N、输出目录、流体网格、流体会话、电网格、电会话**。
预处理用**单进程**运行，不要在外面加 `mpiexec`。

查看每个单元分给了谁：

```bash
cat partitions/partition-map.csv
```

CSV 四列：`region, element_id, incns_rank, adr_rank`。流体单元的 `incns_rank` 与 `adr_rank` 相同；固体的 `incns_rank` 为 `-1`。

### 4.2 启动两个求解器

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpiexec \
    -np 2 "$mhd_fluid_solver" --cwipi fluid "$PWD/partitions/fluid_xml" \
    : -np 3 "$mhd_electric_solver" --cwipi electric "$PWD/partitions/electric_xml" \
    > log.coupled-mhd 2>&1
```

注意：

- 冒号 `:` 是 MPI 的"多程序启动"语法，两段属于**同一个** MPI 作业；
- 输入是 `partitions/fluid_xml` 这样的**目录**，不是原始 XML；
- **不要**开两个终端分别运行两次 `mpiexec`，两边会互相等不到。

### 4.3 检查结果

```bash
python3 "$mhd_case_dir/check_results.py" log.coupled-mhd
```

日志里还应该能找到两次：

```text
Coefficient coupling: ... layout validated once.
```

它表示流体和电两个程序对流体单元的布局检查通过了一次。

---

## 5. 结果在哪里、怎么看

### 5.1 目录结构

```text
student-first-run/
├── fluid-mesh.xml, fluid.xml           输入副本
├── electric-mesh.xml, electric.xml     输入副本
├── log.partition                       预分区日志
├── log.coupled-mhd                     求解日志
└── partitions/
    ├── partition-map.csv               单元 → 进程 对照表
    ├── fluid_xml/                      每个流体进程一份 XML
    ├── electric_xml/                   每个电进程一份 XML
    ├── fluid.fld/                      并行流体场（目录）
    └── electric.fld/                   并行电场（目录）
```

> `.fld` 是**目录**，里面是 `Info.xml` 和各进程数据。复制结果时要复制整个目录，不能只拿一个 `P0000000.fld`。

### 5.2 转成 ParaView 能读的 VTU

```bash
cd "$BUILD/student-first-run"
"$mhd_fieldconvert" fluid-mesh.xml fluid.xml partitions/fluid.fld fluid.vtu
"$mhd_fieldconvert" electric-mesh.xml electric.xml partitions/electric.fld electric.vtu
```

如果提示找不到 `FieldConvert`，先编译它：

```bash
cmake --build "$BUILD" --target FieldConvert -j 4
```

### 5.3 在 ParaView 里检查什么

打开 `electric.vtu`，先看 `sigma`：

- 流体区域应为 **1**；
- 固体区域应为 **0.1**。

再看 `phi` 和 `Jy`：

- 电势在流固界面**连续**；
- 电势的斜率（电场）允许在界面跳变；
- 电流方向应与解析解一致。

打开 `fluid.vtu`，看 `u` 应为抛物线分布，最大约 `0.1`。

---

## 6. 换成自己的算例

### 6.1 四个输入文件

| 文件 | 放什么 |
| --- | --- |
| `fluid-mesh.xml` | 只有流体几何 |
| `electric-mesh.xml` | 流体 + 固体完整几何 |
| `fluid.xml` | 流体方程、变量、展开、边界、参数、耦合模板 |
| `electric.xml` | 电势方程、材料、展开、电边界、耦合模板 |

### 6.2 必须满足的五条规则

1. **几何一致**：电网格的流体部分必须与流体网格单元 ID、顶点 ID、连接、方向完全相同。做法是画一张完整网格，再按原编号提取流体部分。
2. **composite 对应**：`C[0]` 是流体，`C[1]` 是固体；电网格 `DOMAIN` 为 `C[0,1]`，流体网格为 `C[0]`。
3. **展开一致**：两边流体的速度基函数、`NUMMODES`、`NUMPOINTS`、求积点必须相同。例如都写 `NUMMODES="4" NUMPOINTS="5" TYPE="MODIFIED"`。
4. **求解器开关**：`fluid.xml` 里要有 `ElectricPotential="Coupled"` 和 `ForcingLorentz`；`electric.xml` 里 `EQType="CoupledElectricPotential"`，并有 `ELECTRICPOTENTIAL` 段。
5. **改完 XML 必须重新预分区**，并使用新的输出目录。旧分区不会自动更新。

`electric.xml` 的关键片段：

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

`fluid.xml` 的关键片段：

```xml
<EXPANSIONS>
    <E COMPOSITE="C[0]" NUMMODES="4" NUMPOINTS="5"
       FIELDS="u,v,phi,p" TYPE="MODIFIED" />
</EXPANSIONS>
...
<I PROPERTY="ElectricPotential" VALUE="Coupled" />
```

耦合块（`<COUPLING>`）直接保留模板即可，`MHDPartition` 会自动把它改写成 `Coefficients` 模式。

### 6.3 用 `MHD_CASE_DIR` 运行自己的输入

```bash
MHD_CASE_DIR=/绝对路径/my-inputs \
MHD_CHECK_RESULTS=0 \
MHD_TIMEOUT=3600 \
FLUID_RANKS=2 SOLID_RANKS=1 \
    "$mhd_case_dir/Allrun" /绝对路径/my-run-2-1
```

- `MHD_CHECK_RESULTS=0`：自己的物理问题没有配套解析解，跳过内置检查；
- `MHD_TIMEOUT=3600`：大算例要放大超时；
- 输入、输出目录都建议写**绝对路径**，且不要相同。

> 内置的 `ExactSolution` 和 `BodyForce` 是为了验证这个特定解析解。换成真实问题时要删掉或替换，否则会得到"人为维持"的稳态。

---

## 7. 常用参数速查

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `FLUID_RANKS` | 2 | 流体分区数 M |
| `SOLID_RANKS` | 1 | 固体分区数 N |
| `MHD_CASE_DIR` | 本算例目录 | 四个输入 XML 所在目录 |
| `MHD_CHECK_RESULTS` | 1 | 是否检查内置解析解 |
| `MHD_TIMEOUT` | 120 秒 | `mpiexec` 超时 |
| `MHD_TRANSFER` | `Coefficients` | 可改为 `Cwipi` 对比旧点插值模式 |
| `NEKTAR_BUILD_DIR` | 源码下 `build` | 构建目录 |
| `INCNS_SOLVER` / `ADR_SOLVER` / `MHD_PARTITIONER` | 自动选择 | 手动指定程序路径 |

第一个位置参数是**输出目录**。

---

## 8. 常见错误速查

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `Paired fluid layouts differ` | 两套网格的流体单元 ID/顶点/方向/阶数不一致，或改了 XML 没重新预分区 | 从同一张完整网格提取；重跑 `MHDPartition`；换新输出目录 |
| `Coefficient exchange requires rank-matched pre-partitioned meshes` | 直接拿原始 XML 启动 | 先预分区，输入 `partitions/..._xml` 目录 |
| `Unified ADR partitions must be fluid-only on ranks 0..M-1` | M/N 与分区目录不匹配，或用了旧目录 | 用一致的 M、N 重新预分区 |
| `mpiexec` 卡住直到超时 | 两个程序不在同一个 `mpiexec` 里；`-np` 与分区数不一致 | 检查冒号语法和进程数（2M+N） |
| `No rule to make target 'MHDPartition'` | 配置时没开 `NEKTAR_UTILITY_EXTRAS` | 加 `-DNEKTAR_UTILITY_EXTRAS=ON` 重新配置 |
| `CoupledElectricPotential` / `Coefficients` 未注册 | 运行了旧安装或其他构建目录的程序 | 用 `$mhd_*` 变量指向当前 `BUILD` 中的程序 |
| `Unknown parameter` | `PARAMETERS` 中先引用、后定义 | 基础参数写在前面，引用它的参数写在后面 |
| 首次配置 CWIPI/HDF5 下载失败 | 网络中断，留下 0 字节文件 | 删除 `build/ThirdParty/` 中对应文件后重试 |
| 结果里没有 `.fld` 文件，只有目录 | 并行输出本来就是目录 | 保留整个目录，用 `FieldConvert` 读取 |
| 解析检查失败 | 改了物性/几何但解析解没同步 | 同步修改 `ExactSolution`、`BodyForce`、边界；或设 `MHD_CHECK_RESULTS=0` |

---

## 9. 下一步学什么

1. **想理解输入文件每一段**：读 [TUTORIAL_zh.md](TUTORIAL_zh.md) 第 5 节。
2. **想自己画网格**：读 TUTORIAL 第 4 节（Mesh2D `square` 分支 + 网格提取脚本）。
3. **想手动控制每一步**：读 TUTORIAL 第 6 节，或本文第 4 节。
4. **想跑回归测试**（需要配置时加 `-DNEKTAR_BUILD_TESTS=ON`）：

   ```bash
   cmake --build "$BUILD" --target MHDCoefficientTransferCheck Tester -j 4
   ctest --test-dir "$BUILD" \
       -R '^IncNavierStokesSolver_(MHDCoefficients|CoupledMHDThickWall2D)$' \
       --output-on-failure
   ```

5. **想比较两种传输方式**：

   ```bash
   MHD_TRANSFER=Cwipi FLUID_RANKS=2 ELECTRIC_RANKS=2 \
       "$mhd_case_dir/Allrun" "$BUILD/student-point-transfer"
   ```

   `Cwipi` 是旧的逐点插值模式，只用于对比，正式计算建议用默认的 `Coefficients`。

---

## 10. 一句话总结

> **先跑通默认算例，再手动做一遍，最后才改网格和参数。**
> 改任何 XML 之后，记住三件事：重新预分区、换新输出目录、核对总进程数 2M+N。

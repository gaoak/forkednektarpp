# 完整三维薄导电壁方管

本例使用完整三维六面体网格，只运行 IncNavierStokesSolver 的 Internal 电势路径。无需 ADRSolver 或 CWIPI，也无需固体体单元。

流体域为 [0,1]³，网格为 2×2×2 个六面体。流向为 z，z=0/1 周期；x=0/1 是绝缘无滑移侧壁，y=0/1 是薄导电无滑移壁。磁场沿 y，Ha=1，运动黏度 0.1，流体电导率 1，半宽 a=0.5，壁电导比 c=0.05。因此薄壁系数 WallLength=c*a=0.025。使用固定轴向体积力 1 驱动，步长 0.01，共 200 步；这是低成本验证示例，未声称已达到稳态或完成空间收敛。

## 输入格式

```xml
<I PROPERTY="ElectricPotential" VALUE="Internal" />

<R VAR="phi" USERDEFINEDTYPE="ThinConductingWall"
   VALUE="0" PRIMCOEFF="WallLength" />
```

WallLength=sigma_wall*wall_thickness/sigma_fluid，其长度单位与网格坐标相同。R 标签复用边界矩阵的装配入口；ThinConductingWall 实际加入的是面的切向刚度，不是普通 Robin 的边界质量项。

完整三维的 phi 方程加入：

```
integral_Gamma WallLength * grad_s(phi) · grad_s(test) dS
```

grad_s 使用三角形或四边形面的嵌入式几何度量，包括全部物理坐标方向。矩阵装配、静态凝聚、全矩阵 Dirichlet 修正和迭代矩阵乘法采用相同算子。输入中无需设置 Homogeneous；z 方向由真实网格和周期边界表示。

VALUE=0 表示薄壁外表面绝缘。面电流 K=-sigma_wall*t*grad_s(phi) 可在壁内流动，并与流体交换法向电流：取法向 n 指向壁内，有 J_f·n=div_s(K)。暴露边缘的 K·nu_s=0 是自然边界条件，无需独立边缘输入。薄壁面的内部网格边通过 CG 装配保持电势连续与电流平衡；若相邻两块薄壁共用边，它们按理想电接触连接，而非分别绝缘。独立边缘电极、接触电阻和电隔离接缝尚无独立输入模型。

若所有电边界均为绝缘或薄壁，电势仍有加法常数零空间，继续采用现有的参考电势处理。电流不受该常数影响。本例显式指定 SingularVertex=0，便于与截面参考解对齐。

当前 MHD 模型要求固定、无 Mapping 的网格，静止无滑移壁，零外加电场 E0，以及每个边界面上常数、非负的 WallLength。壁厚应小于沿壁电势变化的尺度；该模型并不等价于任意有限厚壁。

## 编译与运行

在仓库根目录：

```bash
module load nektar++/latest
cmake --build build --target IncNavierStokesSolver FieldConvert -j4
bash solvers/IncNavierStokesSolver/Examples/MHDThinWall3D/Allrun
```

默认用两个 MPI 进程，结果在 `build/mhd-thin-wall-3d/`。设置 FLUID_RANKS=1 可串行运行；Allrun 的首个参数可指定新输出目录。输出 `result.plt` 为 Tecplot 双精度二进制，包含 u、v、w、phi、p。选择 w 查看轴向流动。

不用脚本时，在算例副本目录运行（此处为 Debug 构建）：

```bash
mpiexec -np 2 /home/agao/testcode/nektar/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver-g mesh.xml session.xml
/home/agao/testcode/nektar/build/utilities/FieldConvert/FieldConvert-g -f mesh.xml session.xml mesh.fld result.plt:plt:double
```

## 验证与复现

```bash
ctest --test-dir build --output-on-failure -R '^ADRSolver_ThinWall3D'
ctest --test-dir build --output-on-failure -R '^IncNavierStokesSolver_MHDThinWall3D$'
```

标量解析测试包括六面体的静态凝聚、DirectFull、IterativeFull、Xxt/MPI，倾斜拉伸四边形面，倾斜三角形面，以及全闭合薄壁的电势零空间。Dirichlet 混合边界的非零解析电势用于检验全矩阵边界修正；这些标量测试中规定的非零边界载荷是测试用制造解，MHD 示例使用的仍是 VALUE=0。

MHD 测试把相同时间步、驱动力、截面网格和阶次的 3DH1D 结果与完整三维串行、两进程结果逐点比较。比较速度以及移除一个常数电势差后的 phi，同时检查 z 方向不变性。报告和各自的 Tecplot 文件在 `build/tests/mhd-thinwall3d/`。这里是沿 z 均匀的三维解验证；非零表面切向方向及几何度量另外由倾斜面的解析测试验证。

本次验证已通过 8 项新增三维标量测试、3 项原二维/准三维薄壁测试和 4 项已有 MHD 回归。三维 MHD 对照也通过：串行和两进程速度的最大差分别约 2.5e-15、3.1e-9；扣除电势常数差后的最大 phi 差约 9e-7，受 FieldConvert CSV 六位有效数字限制。最终绘图文件均为 Tecplot `.plt`，CSV 和 JSON 仅用于内部检查。

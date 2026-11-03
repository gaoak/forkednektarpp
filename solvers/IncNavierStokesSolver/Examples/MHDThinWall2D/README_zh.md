# 薄导电壁 Hunt 流动：单独使用 IncNSSolver

本例只划分流体网格，不运行 ADRSolver，不使用 CWIPI。流速沿 z，磁场沿 y；上下壁为薄导电壁，左右壁绝缘。流体截面 [0,1]²，半宽 a=0.5，Ha=1，壁电导比 c=0.05。四个流体单元用于演示，不是高 Ha 精度网格。

## 输入

在薄壁对应的 phi 边界使用：

```xml
<R VAR="phi" USERDEFINEDTYPE="ThinConductingWall"
   VALUE="0" PRIMCOEFF="WallLength" />
```

其中 `WallLength=sigma_wall*wall_thickness/sigma_fluid=c*a`，与网格坐标采用同一个长度单位。PRIMCOEFF **不是直接填 c**，除非采用半宽 a=1 的无量纲坐标。本例 WallLength=0.025。

`VALUE=0` 表示壁外表面绝缘。薄壁端点默认绝缘，不需要单独指定端点条件；壁面分成多个边界单元时，内部接点通过全局 CG 装配保持电势连续及壁电流平衡。不要在薄壁同时施加普通 Insulator 或等电势条件。左右侧壁使用 `<N VAR="phi" USERDEFINEDTYPE="Insulator" VALUE="0" />`。

本实现以 R 边界作为输入及装配入口，但加入的是切向刚度，而非普通 Robin 的质量项。电势的常数零空间仍然保留，求解器自动沿用参考电势处理。无需新 EQType 或外部耦合。

## 模型

静止、无外加电场、无接触电阻的薄壁中，面电流 K=-sigma_wall*t*grad_s(phi)。取法向 n 从流体指向固壁，则 J_f·n=div_s(K)。电势方程增加

```
integral_wall WallLength * grad_s(phi)·grad_s(test) ds
```

与流体内部弱电势方程一起求解。外表面和边缘无电流流出；流体可以向壁内输送电流，再从壁的其他位置流回流体。

首版范围：二维流体网格，或 Homogeneous=1D、HomModesZ=2 的充分发展截面；固定、未使用 Mapping 的网格；每个边界单元上常数、非负的 WallLength。MHD 使用 VALUE=0、外加 E0=0，并假定静止无滑移壁。上下壁可以设置不同系数。暂不支持三维壁面、非零 Fourier 模态、外接电路、接触电阻、独立边缘电极或壁内电动势。薄壁近似要求厚度相对于沿壁电势变化尺度足够小。

## 编译、运行与 Tecplot

在仓库根目录：

```bash
module load nektar++/latest
cmake --build build --target IncNavierStokesSolver FieldConvert -j4
bash solvers/IncNavierStokesSolver/Examples/MHDThinWall2D/Allrun
```

默认结果在 `build/mhd-thin-wall-2d/`；Tecplot 文件 `result.plt` 包含轴向速度 w、电势 phi 和压力。设置 `FLUID_RANKS=2` 可做流体 MPI 分区，沿 z 不再分区。Allrun 的第一个参数可指定新结果目录。

不用脚本时，加载 module 后在算例副本目录运行当前 build 的程序（Debug 构建有 -g 后缀）：

```bash
mpiexec -np 2 /home/agao/testcode/nektar/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver-g mesh.xml session.xml
/home/agao/testcode/nektar/build/utilities/FieldConvert/FieldConvert-g -f -m homplane:planeid=0 mesh.xml session.xml mesh.fld result.plt:plt:double
```

这里只输出流体体电势；面电流可由 -sigma_fluid*WallLength*grad_s(phi) 计算，其单位为 A/m，不能直接当作体电流密度 A/m²。

## 验证

`ADRSolver_ThinWallLaplace` 与 `ADRSolver_ThinWallLaplaceFull` 使用精确调和多项式 phi=x²-y²，分别验证静态凝聚和完整矩阵的边界刚度及 Dirichlet 修正。它们是底层矩阵验证；MHD 模型本身仍单独运行 IncNSSolver。

另在 `build/coupled-hunt-ha1/hunt-ha1000-thinwall-model/` 准备了 Ha=1000、c=0.05、G=3379.82 的加密验证算例，与已用的有限厚壁算例采用相同流体网格。该模型应与薄壁极限精确解比较，不应要求其与 h=0.1 的有限厚壁结果相同。

本次已完成加密验证：5750 步达到稳态，Q=3.9998106333，相对 MHDExact 薄壁极限偏差 4.09917e-05%；两条中心线峰值归一化最大误差分别为 0.00103902% 和 0.0650873%。详细说明及 Tecplot 文件在上述加密验证目录。小型示例已验证单进程和两进程运行。

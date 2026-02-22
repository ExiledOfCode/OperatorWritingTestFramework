
# 框架的总体概况：
# test_op_name.sh   中存放的是需要进行测试的算子名称
# tests 是测试google测试框架的地方
# tool 中是检测当前gpu的参数的脚本
# include   关于测试框架的封装
# src
#   gemm
#       hand_idx    中是纯手算idx
#       frame_idx   中是使用封装的框架计算idx
#   reduce
#
#
#
#

# run command
```sh

# 进行算子正确性和速度的测试
./run.sh

# debug的算子测试脚本
./debug.sh

```
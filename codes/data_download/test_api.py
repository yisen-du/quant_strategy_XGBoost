# 用AM系统账号和密码登录

from causis_api.const import get_version

from causis_api.const import login

login.username = "yisen.du"

login.password = "Eason@20010917"

login.version = get_version()
# 导入函数包

import pandas as pd

import numpy as np

from causis_api.data import *

# 数据接口测试

print(get_price('S.CN.SSE.000300', '2020-09-01', '2020-09-22'))
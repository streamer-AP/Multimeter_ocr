# 使用说明

## http 接口服务

### 请求

``` json
{
    "ask":[codec]
}
```

* codec = 0,请求识别结果; codec = 1,请求识别结果结果+debug图片

### 接收

识别结果

```json
{
    "status":[codec],
    "message":"",
    "digit":"",
    "unit":""
}
```

* codec = 0,正常返回; codec=1, 未检测到表; codec =2, 未识别到数字; codec=3, 数字不稳定; codec=4, 特征点不足，建议添加模板; codec=-1，系统未知错误

* message, 返回状态描述语句字符串

* digit, 识别结果字符串

* unit, 识别单位字符串
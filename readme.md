## 说明

这是一共万用表识别程序，可以识别表的读数，单位，正负，小数点等。 目前支持三种型号的万用表。理论上可以支持数码管数字+圆盘旋钮结构的任意数码表。项目目标准确率98%。

![image](https://github.com/streamer-AP/Digital_Panel_OCR/edit/main/tmp.jpg)


## 使用方法

如果使用视频作为输入，请从链接: https://pan.baidu.com/s/1ljYsv9dD624vnbRuRAl1SQ  密码: wdm6 下载，并修改config.json的camera.file字段设为对应的文件名。

如果使用摄像头直接输入，只需config.json camera.file字段设为0即可。

程序提供了摄像头内参的标注工具，参考calibrate.py。使用棋盘格标定。

注意当使用摄像头运行时，为获取最新的实时图片，需对camera.py稍作修改，否则会存在缓存积累的问题。

程序同时提供了调试运行方法，只需运行:

``` sh
python main.py
```

也封装了http服务，基于flask，在config.json中设置IP， 端口即可。运行命令为

```sh
python server.py
```

当需要添加自己的表时，只需运行mask_generator.py程序，分别对表身，表盘，屏幕进行标注即可。按s保存，q退出。 标注结果按示例写入config.json. 

## 技术细节

整体分为两部分：对齐+数字/单位识别。 对齐使用SIFT特征进行，将表与标准表对齐后在固定区域切分出表盘，屏幕。

数字/单位识别各使用一个resnet18进行。

## 运行速度

RTX 3090 15fps

jetson nano 1.5 fps


FROM docker.dm-ai.cn/kg/zhuangjingyu/pytorch_gpu:0.0.4
ADD build /code
RUN pip install loguru
WORKDIR /code
ENV LANG C.UTF-8
#Relation Net ori 5shot 将RN网络的Conv3改为parse，调整了CNN和RN和resnet50

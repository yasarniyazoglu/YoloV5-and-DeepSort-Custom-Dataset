# 경량화 플랫폼 Jetson TX2을 활용한 버스 내부 모니터링 시스템

> ***Jetson TX2에서 YOLOv5 + DeepSORT를 활용하여 버스 내부 다중 영상에 대하여 객체 탐지 및 추적하여 승차 또는 하차 인원 측정 및 사고를 감지하는 시스템입니다***
## 과제 소개
**개별 버스에 설치된 Jetson TX2**를 활용하여 **다중 영상**에 대해 객체 탐지 및 트래킹하여 승차 또는 하차 인원 카운트 및 사고 발생을 감지한다. 클라우드 서비스와 연동을 통해 승차 또는 하차 인원수와 사고 발생시 이후 상황을 보여주는 영상을 특정 시간동안 실시간으로 전송한다.  
저장된 데이터를 활용하여 승객과 관리자의 편의를 위한 인터페이스를 제공할 수 있다.

> Jetson TX2를 활용하여 네트워크 엣지(Edge)에서 객체 탐지 및 트래킹을 진행하여 다중 버스에 대하여 적용이 가능하도록 하였고, 전체 영상 전송 및 저장에 대한 부담을 줄이고 실시간성을 확보하는데 집중하였다.  
또한, 제한된 리소스를 가진 Jetson에서 다중 영상을 처리가 가능하다는 점에서 본 과제는 의미가 있다.

## 과제 배경 및 목적
서울 교통공사는 2017년부터 교통카드 태그 기반으로 시내버스 혼잡도 데이터를 제공하고 있다. 서울특별 시의 경우, 98.9%의 승객들이 교통카드로 버스를 타고 하차할 때는 하차 태그를 하기 때문에, 꽤나 정확도가 높다고 할 수 있다. 그러나, **부산시의 경우 하차 태그 비율은 30%, 대구시의 경우 38%로 하차 시 교통카드 태그가 원활히 이루어지고 있지 않다.** 
또한, **기존에는 블랙박스를 통해 종점에서 녹화된 영상의 처리가 이루어지므로 실시간으로 사고에 대한 대처가 불가능하다는 문제**에 집중하였다.   

> 이를 해결하기 위해 교통카드 태그 방식이 아닌 **기본적으로 버스 내부에 설치된 CCTV를 활용**하여 혼잡도를 측정하고, **실시간으로 사고에 대한 모니터링**을 제공하기 위해 이 시스템을 개발하였다.

## 시스템 구성도
![시스템 구성도](https://github.com/pnucse-capstone/capstone-2023-1-26/assets/100478309/b01d5776-0e0f-4fb8-b591-618572217f6f)  

Jetson에서 RTSP(Real Time Streaming Protocol)를 이용하여 실시간으로 하나의 버스 내부 3개의 CCTV영상을 받아온다. 영상은 입구, 출구와 내부 영상이 존재하고 각각은 승차 인원 측정, 하차 인원 측정 그리고 사고 감지에 사용된다.

**승차/하차 인원수 측정**: YOLOv5와 DeepSORT를 활용하여 승차 또는 하차하는 승객의 인원수를 측정하고 MQTT 프로토콜을 이용하여 전송한다. 클라이언트는 버스 번호로 구분된 토픽을 구독하여 각 버스에 대한 승차 또는 하차 인원수와 사고 발생에 대한 정보를 얻을 수 있습니다.  
![Topic구성](https://github.com/pnucse-capstone/capstone-2023-1-26/assets/100478309/92be4094-012b-4001-b513-2969dffbb1d7)

**넘어짐 감지**: YOLOv5와 DeepSORT를 활용하여 넘어짐을 감지하고 넘어짐 감지 시, HLS프로토콜을 이용한 .m3u8파일과 .ts파일을 5초 간격으로 생성합니다. 사고 이후 총 30초의 영상을 생성합니다. 관리자가 사고 사실을 인지하고 실시간으로 영상을 확인할 수 있습니다.

## 시연 영상
https://www.youtube.com/watch?v=_uKhF7eF_cQ&list=PLFUP9jG-TDp96chsm66TfMPlAJXIt6Gr9&index=26&t=38s

## 설치 방법
1. python 설치 (v3.6 ~ v3.8)
2. 라이브러리 설치
```python
$ pip install -r requirement.txt
```

## 실행 방법

#### 설정
```python
$ touch .env
$ vi .env

$ HLSPATH = "your_local_hls_path/hls"
ACCESS_KEY_ID = 'your_IAM_key_ID' #s3 관련 권한을 가진 IAM계정 정보
ACCESS_SECRET_KEY = 'your_IAM_secret_key'
ENDPOINT = "your_iot_endpoint to publish"
PATH_TO_AMAZON_ROOT_CA_1 = "your_Amazon-Root-CA1.pem"
PATH_TO_PRIVATE_KEY = "your_private.pem.key"
PATH_TO_CERTIFICATE = "your_device.pem.crt" 
BUSNUM = [101] (bus num you interest)
```

#### 실행
```python
version 1
1. RTSP
$ python track.py --source rtsp:your_rtsp_address
2. *.mp4
$ python track.py --source video.mp4

verison 2 - default embeded video
$ python track_.py
```

### GPU 사용 방법
1. 자신의 GPU에 호환되는 CUDA, cuDNN, CUDA Toolkit 설치
2. CUDA 버전에 맞는 torch, torchvision, torchaudio 설치

이후 실행
```python
version 1
1. RTSP
$ python track.py --source rtsp:your_rtsp_address --device deviceNum
2. *.mp4
$ python track.py --source video.mp4 --device deviceNum

version 2
$ python track_.py
```

### 예상 문제
1. torch에서 CUDA 인식을 못하는 문제
    > GPU에 맞는 CUDA를 설치하고 CUDA와 호환되는 torch, torchivsion, torchaudio를 다시 설치하세요!

2. AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
    > File /usr/local/lib/python[3.8]/site-packages/torch/nn/modules/upsampling.py:154, in Upsample.forward(self, input)에서 다음과 같이 설정하세요.
    ```python
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
        #recompute_scale_factor=self.recompute_scale_factor
    )
    ```

3. attributeerror: module 'numpy' has no attribute 'float'.
    > numpy 버전을 더 최신 버전으로 설치하세요. 파이썬과 CUDA 및 파이토치 버전에 따라 적절한 버전이 다를 수 있습니다.

## 라이센스
MIT
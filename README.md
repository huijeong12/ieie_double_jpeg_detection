# **셀프 어텐션과 합성곱 신경망을 이용한 이중 압축 JPEG 탐지**
Self-attention and Convolution for Double JPEG Detection <br>
주저자(공동): 서민균 최희정 양유진

<br><br>

## **Abstract**

 In this paper, we propose a network for double JPEG detection. We generated a new dataset using the quantization tables obtained in [3]. The proposed network consists of 1-dimensional convolutional layers and a transformer encoder[6]. We extract DCT coefficient histogram and quantization table from previously mentioned dataset, and utilize them as input to the network. Also, we localize the forgery region—single JPEG block—by detecting if the JPEG block has been compressed more than once or not. This proposed approach achieved higher accuracy than baseline [3] in detecting single JPEG.

 <br><br>

## **Double JPEG Detection**
![이중 압축 JPEG의 DCT 계수 히스토그램 분포](https://github.com/huijeong12/ieie_double_jpeg_detection/blob/main/images/dct-histogram.png?raw=true)

- Single JPEG: Gaussian 분포를 따름
- Double JPEG: 첫번째 압축에서의 Quality Factor를 Q1, 두번째 압축에서의 Quality Factor를 Q2라고 할 때,
    - Q2 > Q1: Periodic Missing Values
    - Q1 < A2: Periodic Peaks and Valleys
    - Wang, Q. et al.(2016). Double JPEG compression forensics based on a convolutional neural network. EURASIP Journal on Information Security, 2016(1), 1-12.
- 이러한 패턴의 차이를 통해 특정 JPEG 이미지 블록이 단일 압축인지, 이중 압축인지 판별하는 방법을 **<i>Double JPEG Detection</i>**

<br><br>

## **Proposed Network**
![네트워크 구조](https://github.com/huijeong12/ieie_double_jpeg_detection/blob/main/images/%08network-architecture.png?raw=true)

<br>

![Localizing Forged Region](https://github.com/huijeong12/ieie_double_jpeg_detection/blob/main/images/localizing-forged-region.png?raw=true)

<br><br>

## **Localization**
![Localization 예시 1](https://github.com/huijeong12/ieie_double_jpeg_detection/blob/main/images/ex-localization1.png?raw=true)

<br>

![Localization 예시 2](https://github.com/huijeong12/ieie_double_jpeg_detection/blob/main/images/ex-localization2.png?raw=true)
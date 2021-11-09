# Abstract
- YoLo의 unified architecture 너무 빨라
- 초당 45 frame으로 이미지를 처리한다.
- 다른 모델들보다 mAP 2배 이상 나와 (당시)
- YOLOv1은 localization error가 좀 있지만 배경에대한 false positive는 적은 경향이 있다.
- Single Network로 end-to-end optimization했다.

# 1. Introduction
- RCNN이나 다른 모델들의 Detection을 위한 대처는 복잡한 Pipeline을 가지고 있다. 그래서 최적화 하기도 어렵고 느린거다.
  - 게다가 Region Proposal, Classification 등을 분리해서 학습시킨다..;
- YoLov1 Object Detection을 단일 Regression 문제로 재설정했다.
  - image pixel들에서 바로 bounding box 좌표, Class 확률을 얻어낸다.

![image](https://user-images.githubusercontent.com/69780812/140860260-820d6023-a946-43a6-9720-f94d1381b584.png)
- Signle Convolutional Network은 다양한 Bounding Box들을 Predict하고 이 Box들에서 Class 확률을 Predict한다.
- YoLo는 전체 이미지로 Train 하면서 Detection Performace를 최적화해나간다.

## 장점
1. YoLo는 매우 빠르다.
- 45fps (Titan X GPU)
- fast version은 150fps도 나온다고한다.
- 또한, YoLo는 다른 real-time system들보다 2배 이상의 mAP를 얻었다.

2. Full Image를 학습하고, 추론한다.
- Sliding Window (X), Region Proposal-based Technique (X)
- Fast R-CNN은 더 큰 context 정보를 얻을 수 없었기 때문에 이미지에서 Background Error가 많았다. 하지만 이와 비교해보면 YoLo는 Background Error가 절반 이하다.
> background Error : 배경 이미지에 객체가 있다고 탐지한 것

3. YoLo는 Object의 gerneralizable representation들을 학습한다.
- 새로운 도메인에 적용해봐도 잘 되더라
- 예상치 못한 input에서도 잘 되더라

# 2. Unified Detection
- 당시 분리되어 있던 Object Detection의 Component들.. 하나의 Neural Network로 통합했다는 의미다.
  - 이로 인해 bounding box를 predict하면서 class도 predict할 수 있게 됐다.
  - 또한 Full Image에 대해 전체적으로 추론하면서 모든 Object를 추론해나가게 됐다는 것이다.


![image](https://user-images.githubusercontent.com/69780812/140862279-7b445f73-a2a5-4880-8658-d53a9027f2fd.png)
- Input Image를 SxS Grid로 나눈다.
  - Object의 center가 grid cell에 맞아 떨어지면 그 grid cell은 Object를 Deteting한 것이다.
- 각 grid cell은 Bbox와 그 Bbox의 confidence score를 predict한다.
  - confidence score : 얼마나 model이 그 box를 포함하고 있는지에대해 믿을만한지에 대한 것이다. (객체를 갖고있냐에 대한 믿음)
  - 또한, box가 실제 object를 predict하고 있는지에 대한 정확도이다.

![image](https://user-images.githubusercontent.com/69780812/140863963-9598dafc-9572-453d-a3fd-30013358623d.png)
- YoLo v1에서는 confidence를 위와 같이 정의하고 있다.
- Cell에 객체가 존재하지 않으면 Pr(Object) = 0, 즉 confidence = 0이 된다.
- 객체가 존재하면 1이되고 IoU와 곱해져 Confidence는 IOU가 된다.
---
- bounding box는 5개의 prediction들로 구성된다. : x, y, w, h
  - (x, y) : grid cell 경계선에 대한 box의 상대적인 center 좌표이다.
  - (w, h) : width, height
  - confidence : IoU(predict box & ground truth)

![image](https://user-images.githubusercontent.com/69780812/140864483-818058cd-31e2-4796-8073-33d11cb109c1.png)
- grid cell은 class를 조건부 확률(C)로 예측한다.
- 위 확률들은 grid cell에 object가 포함되어있다는 전제하에 나온 것이다.
- 이러한 Prediction들은 S x S x (B*5 + C) tensor로 encoding 된다.
  - S : grid cell num
  - B : Bounding box (Bounding Box num)
  - C : Conditional Probability Num (ClassNum)
- grid cell 당 class probability들에 대한 set를 predict하게된다. (Bbox 갯수에 상관 없이)

![image](https://user-images.githubusercontent.com/69780812/140866902-64335243-8b95-43bd-b26c-806a316c24ce.png)
- Test 시에는 조건부 class 확률을 곱해준다.
- 그리고 개개의 box confidence predction을 곱해준다.
- 이것은 각 box에 대한 특정 class confidence를 얻게 되는 것이다.
- 이 score는 box안에 해당 Class가 존재할 확률과 얼마나 그 box가 Object에 잘 맞춰져있는지 둘다 encoding 한다.

# 2.1 Network Design
![image](https://user-images.githubusercontent.com/69780812/140864689-7843e35f-40a5-41e4-86a4-ce3787683eec.png)
- GoogLeNet model 영향을 받아 구조를 설계했으며 1x1 dimension reduction 수행 후 3x3 conv를 수행한다.

# 2.2 Training
- Final layer는 class 확률과 bbox 좌표를 predict한다.
- bbox의 width, height를 normalize한다.
  - 0 ~ 1 사이 값을 갖게 된다.
- Bbox x, y좌표를 특정 grid cell에 위치시키기 위해 offset으로 매개 변수화 한다.
  - 0 ~ 1 사이로 값을 bound한다.
- YOLO는 grid cell당 많은 bbox를 predict한다.
- 학습 시 오직 하나의 Bbox를 얻기 위해 각 box predictor에게 object를 predicting할 "책임"을 assign 했다.
  - 현재 가장 높은 IOU를 가지고있는 prediction을 기반으로 했다.


## Loss Function
- sum_squared error를 최적화하며 **여러 문제**가 있어 수정한다.
- localization error와 classification error를 동일하게 가중치르 두는 문제
  - 많은 grid cell이 어떠한 Object도 포함하지 않는다. 
  - 이는 그런 Cell들의 confidence score를 0으로 Push하게 만들고 Object를 포함한 cell의 gradient를 더 강하게 만들어줄 것이다.
  - 이런것 때문에 모델이 불안정해지고 학습 시 발산하게 되는 요인이된다.
  - 그래서 **bbox 좌표 prediction에 대한 loss를 증가**시키고 **object를 포함하지 않는 box들의 confidence prediction loss를 감소**시켰다.

![image](https://user-images.githubusercontent.com/69780812/140867942-ca7b5ee5-91c4-4c10-925c-c4f40e069287.png)
- 위를 수행하기 위한 두가지 파라미터다.
- lambda_coord는 Bbox 좌표 손실에 대한 파라미터다.
  - 높은 패널티 필요
- lambda_nobj는 객체를 포함하고 있지 않은 박스에 대한 Confidence 손실에 대한 파라미터다.
  - 배경의 경우 0.5 가중치를 두어 패널티를 낮췄다.
- 위 두가지 파라미터로 localization error(bbox 박스 좌표)와 classification error(잘못된 클래스 분류)에 가중치를 다르게 부여한다.
---
- 또한 작은 박스와 큰 박스에 대해서도 다르게 가중치를 뒀다.
- 더 큰 box의 작은 편차가 작은 box의 작은 편차 보다 덜 중요하다는 것을 반영했다.
  - bbox의 width, height에 squared root를 씌워 이를 predict하도록 했다.

![image](https://user-images.githubusercontent.com/69780812/140869731-36829aa8-b529-426a-8c0b-96f6e08405a7.png)
- grid cell에 object가 존재한다면 classifiaction error에 penalty를 부여한다는 것을 명심하자.
  - 조건부 확률이 의논된 이유
- 그리고 만약 predictor가 ground truth box에 대한 책임을 갖는다면 bbox 좌표에 penalty를 준다.
  - grid cell에서 가장 높은 IOU를 갖는 predictor에 해당하는 말이다.

---
![image](https://user-images.githubusercontent.com/69780812/140869762-c308b653-8167-405f-acda-cf68ed7fba2b.png)
- 만약 cell i에 object가 나타나면이라는 denote

![image](https://user-images.githubusercontent.com/69780812/140869849-13e7fcd7-ffd0-4612-9596-e8add47980f0.png)
- cell i에있는 j 번째 bbox predictor는 그 prediction에 대한 책임을 갖는다는 것은 denote한다.
- 객체가 감지된 Bbox를 말한다.(가장 높은 IOU를 갖고있는 Bbox)
---
![image](https://user-images.githubusercontent.com/69780812/140871185-9edf111d-c6ca-411a-a458-a2d78e27c76b.png)
- x, y는 바운딩 박스 좌표
- 정답 좌표와 예측 좌표 차이를 제곱해 error를 계산한다.
- obj_1_ij는 i 번째 grid에서 j번째 객체를 포함한 바운딩 박스를 의미한다.
  - 여기서 obj는 객체가 탐지된 Bbox를 의미하며 가장 높은 IOU를 갖고 있다.
- lambda_coord는 localization error의 페널티를 키우기 위한 파라미터이다.
  - 5를 지정하면 객체를 포함한 Bbox의 좌표에는 5배의 페널티를 부여하는 것이다.

![image](https://user-images.githubusercontent.com/69780812/140871362-b1160aff-faa1-4b0e-89ea-6a483f004785.png)
- 객체를 포함한 Bbox에 대한 Confidence error이다.

![image](https://user-images.githubusercontent.com/69780812/140871445-3fa0e00e-5202-48c2-951e-0491b4f3951d.png)
- 객체를 포함하지 않은 Bbox에 대한 Confidence error이다.

![image](https://user-images.githubusercontent.com/69780812/140871511-7dddfa33-66ce-4e03-a7f1-cdc32551401b.png)
- p(c) : Class Probability
- 클래스 확률에 대한 Error이다. 객체를 포함한 Bbox에 대해서만 계산한다.
- Classification Error에 대한 부분으로 볼 수 있겠다.

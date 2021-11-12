# 1. Intro
- 약간의 변화를 줬더니 YOLO의 성능이 좋아졌다.

# 2. The Deal
## 2.1 Bounding Box Prediction
- anchor box들을 dimension cluster를 사용해서 predict하여 사용했었다.

![image](https://user-images.githubusercontent.com/69780812/141421054-01cde75a-9592-4272-aad5-0e34a203a9d0.png)
- Network은 4개의 coordinates를 predict 한다.
  - t_x, t_y, t_w, t_h
- c_x, c_y : top left corner로부터 offset. 즉, cell의 좌상단 위치
- p_w, p_h : bounding box prior의 width, height
  - 여기서 bounding box prior은 anchor Box를 의미한다.
- Training시 Squared Error loss의 합을 Loss functionㅇ로 사용했다.
  - (t^ - t) : ground truth의 값과 prediction 값의 차이로 gradient를 구했다.
  - groud truth 값은 위의 식으로 변환해서 쉽게 구할 수 있다.

![image](https://user-images.githubusercontent.com/69780812/141421373-c15581d5-72fd-4486-819b-5bed103358ea.png)
- x, y에 대한 prediction 시 grid cell안의 0 ~ 1 위치를 갖는 것으로 학습하고 prediction 한다.
- w, h는 anchor와 target w, h에 대한 log scale을 이용한다.
- x, y offset prediction을 통해 학습시켜보니 잘 안됐다고 한다. 이러한 Scaling이 필요하다는 이유가 되는 것 같다.
- YOLOv3는 objectness score를 logistic regression을 사용해 predict했다.
  - 여기서 최종 bounding box prior는 bounding box prior들 중 ground turth object랑 가장 겹치면 confidence score가 1이 되어야한다.

  ![image](https://user-images.githubusercontent.com/69780812/141430536-98a9dbb2-bac8-4465-9a3a-eb79d4740a56.png)
  - YOLOv3는 objectness score가 가장 큰 anchor box를 only one bounding box prior로 쓰고자 한다.
- 만약에 최종 bounding box prior이 best가 아니지만 어떤 threshold 이상 ground turth object와 겹치면 Faster R-CNN 방식을 따른다.
  - threshold : predction을 무시해도 되는 임계값 (IOU)
  - Faster R-CNN 방식 : 특정 threshold를 넘는 anchor box를 선택한다. 여러 anchor box가 선택될 수 있다는 것이다.
- threshold는 .5를 사용한다.
- 하지만, Faster-RCNN과 다르게 **YOLO는 각 ground truth object마다 하나의 bounding box prior을 assign한다.**
- bounding box prior이 ground truth object에 assign되지 않으면
좌표나 class prediction에 대한 loss가 없도록 한다.
  - 오직 objectness에서만 loss가 발생된다. 

## 2.2 Class Prediction
- 각 bounding box는 mutlilabel classifiaction을 통해 class를 classification 한다.
- class-prediction 시 softmax를 사용하지 않는 것이 더 좋은 Performance가 나서 binary cross-entropy loss를 사용한다.
  - 클래스가 80개 있으면 80개 각각 logistic regression을 적용한 것
- 복잡한 도메인으로 이동할 때에도 이는 도움이 된다고 한다.
- softmax는 정확하게 하나의 클래스를 가지고있다는 가정을 강요한다.
  - 사실 그런 Case는 많이 없다는 것이다.
  - 많은 데이터셋에서 label들이 많이 겹치기 때문이다. (Woman, Person 등)
- Multilabel 접근이 data를 더 좋게 모델한다.
> Multu-label 문제 : 한 image에 객체가 2개 이상인 것 [1, 0, 1]

## 2.3 Predictions Across Scales
- feature pyramid network와 비슷한 컨셉을 사용해서 여러 sclae에서 feature를 추출한다.
- YOLOv3는 3가지 다른 Sclae에서 box들을 predict했다.
- feature extractor는 bounding box, objectness, class predictions를 포함한 3-d tneosr로 encoding되어 predict한다.
- YOLO의 실험에서는 각 sclae에 대한 3 box들을 predict한다.
  - tensor는 4개의 bounding box offset에 대해서 N x N x \[3*(4+1+80)] 이다.
  - 1은 objectness prediction이고 80은 Class predictions다.
  - COCO dataset으로 실험했다.
- 2 layer 이전의 feature map을 2배 upsample 한다.
  - upsample된 feature는 concat해서 합친다.
- 이러한 방법은 upsampled feature에서 좀더 의미있는 정보를 얻게해줬고, 그 이전 Feature map에서 정교한 정보를 얻어 낼 수 있게해줬다.
- final scale(3rd scale)은 이전 계산된 모든 것뿐 아니라 정교한 Feature들을 얻는 이점을 얻게 된다.
- YOLOv3에서는 여전히 k-means clustering으로 bounding box prior들을 결정한다.
  - k = 9 && scale = 3으로 선택해서 sort한다.
  - COCO dataset은 (10x13), (16x30), (33x23), (30x61), (62x45), (59x119), (116x90), (156x198), (373x326)으로 Cluster 되었다.
  - small, medium, large 별로 나누어 anchor를 구성했다.

## Feature Extractor
![image](https://user-images.githubusercontent.com/69780812/141427429-2b677207-7fc7-4992-873c-2d57e33b2d19.png)
- YOLOv3는 Darknet-19말고 새로운 Feature Extractor를 도입했다.
  - Darknet-53
  - Residual이 추가된 것으로 보인다.
- Darknet-19보다 빠르고 강력하고 ResNet-101 or Resnet-152보다도 충분했다.

![image](https://user-images.githubusercontent.com/69780812/141427675-44827336-9c63-46f7-a726-3d8283b70a9c.png)

![image](https://user-images.githubusercontent.com/69780812/141437410-4a44c895-7a01-4827-82b5-b9005afa8376.png)
- YOLOv3의 Architecture로
- 앞 단까지가 Darknet-53이다.
- 그리고 Feature Pyramid 개념을 적용했다.
  - 기존 SSD와 같은 모델에서는 Low단의 feature가 아직 제대로 생기지 않아 bounding box 정보를 제대로 추출하지 못하는 상황이었다.
  - Feature Pyramid는 이를 보완하기 위해 다시 upsample 하면서 Semantic한 정보를 남기면서 shortcut connect로 위치에 대한 정보를 보충하였다. 이러한 정보들을 그대로 YOLOv3는 활용한다.
  - 3가지 scale에 대한 처리를 진행한다.
  - 맨 첫번째 : Larger Object
  - 두 번째 : Medium Object
  - 마지막 : Small Object
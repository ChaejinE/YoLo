# Better
- YOLOv1에서는 Fast-RCNN과 비교했을 때 localization Error가 좋지 않았따.
- 당시 Trend : Deeper, Larger Network..
  - Network가 깊어지고 커지거나 Assemble 형태여야 잘되는 것으로 보였다.
- YOLOv2는 여전히 빠르지만 정확한 detector가 되려고했다.
  - Network가 Scale up 되는 대신에 구조를 간단하게 만들었다.
  - 그랬더니 학습도 쉽게 잘됐다고 한다.
- YOLO의 Perforamce 증가를 위해서 다양한 idea들을 pool했다.
  - Batch Normalization
  - High Resolution Classifier
  - Convolutional With Anchor Boxes
  - Dimension Clusters

## BatchNormalization
- 다른 Regularization을 빼면서 BatchNormalization을 도입했다.
- mAP 2% 가 improvement 됐다.
- model에서 Overfitting 없이 Dopout을 제거할 수 있었다.

## High Resolution Classifier
- 해상도가 맞지 않으면 train 시 input resolution을 조정해야했다.
- high resolution 분류 network를 구성해보니 mAP가 4% 증가하는 결과를 가져다 줬다.

## Convolutional with Anchor Boxes
- Faster RCNN에서는 RPN이라고하는 Region Proposal Netwokr을 사용했다.
  - Faster RCNN에서는 prior anchor box들이 있었다.
  - 이를 위해 YOLOv2는 fc-layer를 제거하고 bbox predict를 위해 anchor box를 사용하게 된다.
- feature map 에서 모든 location에서의 offset들을 predict했다.
  - 좌표를 predict하는 것보다는 offset을 predict 하는 것이 netowrk 학습관점에서 문제를 더 단순화 할 수 있고 쉽게 학습한다고 한다.
- YOLOv2는 모든 FC-layer를 제거한다.
- bounding box predicting을 위해 anchor boxe를 사용한다.
- 하나의 Pooling layer 제거
  - Convolitional layer의 출력의 해상도를 높였다.
- 입력 이미지 크기를 448 에서 416으로 변경
  - feature map에서 location 수가 홀수가 나오기를 원했기 때문
  - 448x448에서 downsample 시 -> 14x14, 416x416 -> 13x13 feature map이 얻어진다. (홀수면 Single center가 생긴다.)
  - 보통 물체는 **(특히 큰 Object)는 이미지의 중앙에 있는 경우가 많다.** output feature map이 ***홀 수 인 것이 좋다.***
  - Why? 짝수로 설정하면 중앙에 4개의 grid cell이 인접한다.
- 기존 YOLOv1에서는 grid cell 하나마다 class를 predict했는데, anchor box를 도입해서 각 anchor box마다 class를 predict하고자했다. 
- anchor box를 이용하면 ***mAP는 감소하고 Recall이 증가***한다. (개선의 여지는 있음)
  - anchor box 사용하지 않았을 때 69.5 mAP와 81% recall
  - anchor box 사용할 때 69.2 mAP와 88% recall을 얻었다.
  - recall이 더 높다는 얘기는 detection을 더 잘하고 있다는 것이므로 성능 개선의 여지가 높다는 것을 의미한다.

## Dimension Clusters
- Anchor box를 수동으로 대략적으로 설정해서 사용하는데, 이는 비효율적이다.
  - 만약 사전에 더 좋은 anchor box 형태로 조정하면 학습이 더 잘이뤄질 수 있다.
- K-means clustering을 통해 training set에 대한 bounding box들을 사전에 좋은 것들을 찾는다.
  - **어떤 aspect ratio가 좋은지에 대해 찾는 과정이다.**

![image](https://user-images.githubusercontent.com/69780812/141072276-6823fcc7-09fb-4f82-a9dd-497012d65381.png)
- k-means의 Euclidean distance를 사용한다.
- standard의 k-means(with 유클리디안)은 큰 박스들에 대해서 작은 박스들보다 더 많은 에러가 생긴다.
  - ground truth bbox와 중심 좌표 거리가 가장 짧은 것을 기준으로 anchor box를 선정하면, IoU가 낮음에도 anchor box가 선정될 수 있기 때문이다. (큰 박스는 더 그럴 것이다. 겹치는 건 많지만, 중심좌표가 더 멀수도 있기 때문이다.)
  - YOLOv2 팀은 좋은 IOU score로 이끌어줄 anchor 박스가 필요했다. 이는 Box의 Size와는 무관하게 독립적인 것이다.

![image](https://user-images.githubusercontent.com/69780812/141073102-97fb8174-8d5a-494e-b24a-c7eed685fa35.png)
- 따라서 distance method를 위와 같은 것으로 정의하게 됐다.
- 그 이전 그림을 봤을 때, 높은 k일 수록 높은 정확도를 얻을 수 있지만 속도가 느려지므로 적절한 k=5값을 선택하게 된다.
- 논문 내용상으로 추정하자면, training data의 ground truth box를 넣어서 k-means clustering을 진행하여 각 centroid 마다 좌표 상관없이 특정 scale, aspect ratio를 가진 ground truth box와 IOU를 계산하여 군집화를 진행한다고 추정된다.
  - centroid가 어떻게 계산되는지는 모르겠지만, k-means 특성상 평균 값이므로 height, width가 그 군집의 평균인 것이 centroid가 되지 않을까 싶다.

![image](https://user-images.githubusercontent.com/69780812/141073483-d551dda4-1fc3-40ef-9a49-ffc1347cc247.png)
- hand-picked, 수동으로 대략 선정했던 것들보다 clustering을 했을 때 더 좋은 결과를 가져온다는 것을 확인할 수 있다.

## Direct location prediction
- anchor box 도입 시 문제 : model instability (모델 불안정)
  - 특히 초기 iteration에서 모델이 불안정한 현상이 있다고 한다.
- 이 불안정 문제는 대부분 box의 (x, y) location을 predicting 할때 발생했다.

![image](https://user-images.githubusercontent.com/69780812/141389360-87f0c038-5345-449e-b3bb-0f0e9a4c2d52.png)
- 기존 Faster RCNN의 bbox regression에서는 bbox가 image상 어떤 곳이든 나타날 수 있는 문제가 있다.

![image](https://user-images.githubusercontent.com/69780812/141219232-84117d44-897b-45e6-928a-351e22d8ce0d.png)
- t_x = 1 : box를 anchor box의 너비를 기준으로 오른쪽으로 이동한다.
- t_x = -1 : box를 anchor box의 너비 기준으로 왼쪽 이동시킨다.
- region proposal network는 t_x와 t_y 값들을 predict한다.
  - x, y는 center 좌표로 위 처럼 계산된다.
- 위 공식은 제약이 없어 모든 anchor box가 이미지의 어느 지점에나 있을 수 있다.
- random initialization은 모델이 **민감할 수 있는 offset들을 predicting하기위한 안정성을 갖도록 하는데 오랜 시간이 걸리게한다.**
- 그래서 grid_cell의 location에 대한 상대좌표를 predict 한다.
  - 이는 ground truth가 0 ~ 1사이로 되도록 해준다.
  - **logistic activation을 사용**해서 netowkr의 prediction을 이 범주(0~1)로 만들도록 제한해준다.
- 위와같은 기존 network는 output feature map의 각 cell에서 5개의 bbox를 predict한다.
  - netowrk는 각 bounding box에대한 **t_x, t_y, t_w, t_h, t_o 5개의 좌표들을 predict**하는 것이다.

![image](https://user-images.githubusercontent.com/69780812/141220998-6dba5bc0-a402-4b2b-a290-985c811ed006.png)
- cell이 image의 top_left corner에서 offset이 c_x, c_y이고, bbox가 p_w(width), p_h(height)를 갖는 다면 prediction은 위와 같이 진행된다.
  - c_x, c_y : 좌측 상단에서 grid cell의 x, y좌표
  - sigma : sigmoid activation function
  - 이 sigmoid(logistic activation function)덕 분에 각 cell 안에 x, y 좌표가 있도록 학습 시킬 수 있게 된다.
- YOLOv2는 위치 prediction을 제한하고 있기 때문에, 이 매개변수들은 학습하기가 더 쉬워진다.
  - network를 더 안정시켜준다.
- bbox center를 predicting하면서 dimension cluster를 사용하는 것이 YOLOv2를 anchor box를 사용한 버전보다 거의 5% 더 좋았다.

![image](https://user-images.githubusercontent.com/69780812/141221616-a3ac9a01-5572-42ee-aed8-c7fbd0e73f6d.png)
- cluster centroid에서 offset들로서 box의 height, width를 predict하는 것이다.
- sigmmoid function을 사용해서 필터 적용 위치를 기준으로 상자의 중심 좌표를 예측한다.
- b_x, b_y, b_w, b_h는 bounding box의 x,y 좌표와 width, height다.
- c_x, c_y : cell의 좌측 상단 x, y 좌표 값
- p_w, p_h : anchor box의 width, height

## Fine-Grained Features
- YOLOv2는 13x13 feature map을 perdict한다.
  - 이는 큰 Object에서는 충분하지만 작은 Object에 대해서 더 정교한 feature를 얻어내는 것이 좋을 것이다.
- Faster-RCNN, SSD 모두 proosal network에서 다양한 resolution으로부터 다양한 feature map을 뽑아냈다.
- pass through layer를 추가해서 26x26 resolution을 갖는 앞단의 layer로 부터 feature들을 가져오도록 했다.
- pass through layer
  - ResNet의 identity mapping과 비슷하다.
  - low resolution feature들과 higher resolution feature들을 concat한다.
  - higher resolution feature를 인접한 Feature들을 채널 방향으로 쌓는다. (spatial location이 아니다.)

![image](https://user-images.githubusercontent.com/69780812/141226155-d9c857ad-182d-442e-8ae4-bd0e4dd3888e.png)
- pass through layer로 인해 26x26x512 featuremap이 13x13x2048 featuremap으로 바뀐다. (분해한 것)
  - detector는 이런 확장된 Feature map을 통해 정밀한 feature들에도 접근할 수 있게 해준다.
  - 이로 인해 1% performance 증가가 있었다고 한다.

## Multi-Scale Training
- YOLOv2는 모델이 다른 size의 image들도 학습하길 바랬다.
  - 기존 448x448 에서 anchor box 추가와함께 416x416로 바꿔서 fine tuning 헀었다.
  - 하지만, conv, pooling layer로 모두 바꿔서 다양한 이미지를 input으로 받을 수 있다.
- input image size를 고정하기 보다 10 batch마다 랜덤하게 새로운 image dimension size를 선택한다.
- YOLOv2는 1/32로 Downsample 하기 때문에 {320, 352, ..., 608}로 다양한 size를 선택했다.
- 이러한 정책은 다양한 input dimension으로 잘 predict하도록 학습시켰다.
  - 이는 다른 resolution에서 detection들을 predict할 수 있다.
- low resoltion에서 YOLOv2는 cheap하고 꽤 정확한 detector다.
  - 288 x 288은 90fps이상 이었고 mAP는 Fast R-CNN 만큼 좋았다.

![image](https://user-images.githubusercontent.com/69780812/141231523-dea34c93-52cd-4bc4-a6f1-042d5bdcd8df.png)
- hgih resolution YOLOv2 모델은 SOTA 78.6mAP를 찍었었다.
  - 여전히 real-time speed였다.

# 3. Faster
- YOLO framework은 GoogLeNet architecture를 기반으로한 custom network을 사용한다.
  - 이 netwokr는 VGG-16보다 빠르다. (정확도는 조금 떨어짐)

## Darknet-19
- YOLOv2의 **classification model**로 사용되는 Nework이다.

![image](https://user-images.githubusercontent.com/69780812/141232122-75864ff9-da79-470b-8add-a6091e8d89ae.png)
- 3x3 filter를 주로 사용 (VGG model과 비슷)
- NIN : prediction을 위해 golbal average pooling
- 1x1 filter : 3x3 conv 사이 feature 차원 압축을 위해 사용
- batch normalization : Training 안정화, 수렴 속도 향상
- ImageNet에 대해 top-1 accuracy가 72.9%, top-5 accuracy가 91.2%를 달성한바 있다.

# 4. Stronger (YOLO9000 내용)
- Classifiaction 및 Detection을 연결해서 습하는 메카니즘을 제안한다.
- detection-specific information을 학습하기위해 bbox coordinate, objectness, 어떻게 Common Object로 분류할지와 같은 label된 image data를 사용한다.
- Trainin하는 동안 detection과 classification dataset 둘다 image를 섞는다.
- detection을 위해 label된 image에대해서 full YOLOv2 loss 를 역전파 시킨다.
- classifiaction image 에서는 Architecture에서 classification에 대한 part에만 loss를 역전파한다.
- Detetcion dataset에서는 흔한 object와 일반적인 label들이 있다.
  - "dog", "boat" 등
- Classifiaction dataset에서는 더 넓고 깊은 범주의 label들이다.
  - "Norfolk terrier", "Yorkshire terrier" 등
- 만약 이 dataset 둘다 학습시키길 원한다면, 이 label을 합칠 논리적 방법이 필요하다.
  - softmax layer ?
  - softmax는 class가 상호 독립적이라느 가정하에 만들어졌다.
  - "Dog", "Norfolk terrier"는 전혀 독립적이지 않다. (문제점)
  - Dataset을 한번에 묶을 수 없다.
- 상호적으로 독립적이지 않다고 가정하는 데이타셋 혼합을 위한 **Multi-Label** model을 사용하게된다.
- 우리가 알고있는 data에 대한 모든 구조들을 싹 다 무시할 것이다.
  - COCO의 class들이 상호 연관성이 아예없다 라는 등의 사전지식을 무시

## Hierarchical classification
- WordNet : concept들을 구조화하고, 어떻게 연관성이 있는지를 보여주는 language database다.
  - directed graph다.
  - *terrier는 dog으로도, canine, domestic animal 등으로 여러 분류로도 나뉠 수 있지만 모든 분류에 대해 graph를 만들어 봤을 때, 짧은 path가 나온 것으로 WordNet을 구성했다고 한다.

![image](https://user-images.githubusercontent.com/69780812/141236357-77e57fc4-ee5b-4cda-85d1-08f14a5711ce.png)
- ImageNet에서 concept들로 부터 계층 Tree를 구성하여 문제를 간단화 한다.
- Final result : Word Tree
- 색칠된 것들이 실제 Label들이다.

![image](https://user-images.githubusercontent.com/69780812/141391312-fc3c38b1-9d36-4e84-952e-b205068bcdbe.png)
- WordTree로 classification을 수행하기 위해 모든 노드에 대한 조건부 확률을 predict한다.
- 위는 "terrier" node 에서의 예이다.
- 특정 노드에 대한 절대적인 확률을 계산하고자한다.
- 간단하게 root에서 tree를 통해 조건부 확률을 곱해간다.

![image](https://user-images.githubusercontent.com/69780812/141236867-1718739c-4c7f-4fa5-a50a-26cee9fada95.png)
- Norfolk terrier에 대한 예이다.
- YOLOv2의 objectness predictor는 Pr(physical object) 값을 준다.
- detector는 bbox와 확률에대한 tree를 predict한다.
  - tree를 통해 타고들어가서 가장 높은 confidence path를 얻는다.
  - 타고들어갈때 어떤 threshold에 도달할 때까지 타고들어간다.
  - 그것이 object의 class를 predict한 것이다.
---
![image](https://user-images.githubusercontent.com/69780812/141236989-2a743e84-62e8-4cfe-b1b9-8f0edb76f289.png)
- classification에 대한 것은 image가 object를 포함하고 있다는 전제하에 진행한다.

![image](https://user-images.githubusercontent.com/69780812/141244724-9e730bbb-48d9-4700-a50d-9b6e74d76f5d.png)
- 학습 진행 시 WordTree1k와 같은 Multi-label을 이용한다.
  - 학습 시 ground truth label을 상위로 Propagation 시킨다.
  - ex) Norfolk terrier가 있으면 상위로 가면서 dog, mammal 등이 있을 텐데 이에 대해 label을 1로 두고 Propagate한다. (Multi label 처럼 되는 셈)

- 같은 색깔을 갖는 부분들이 각각의 softmax에 적용되는 mutually exclusive(상호독립)인 부분으로 WordTree에서 같은 단계에 존재하는 노드들이다.
  - 같은 Level의 것들 끼리 softmax를 취해 predict한다.
  - 그리고 점점 상위로 가면서 각자의 상위 Level Node끼리 softmax를 취한다. (이렇게 구했던 것들로 Conditional Probability를 구해낸다.)
- 이를 이용해 나온 조건부 확률들의 곱을 이용해 확률을 구해 학습을 수행한다.
- WordTree 이용방법은 ImageNet 데이터 뿐아니라 COCO와 같은 다른 데이터들과 결합할때도 사용가능하다.
  - COCO : general concepts
  - ImageNet : Specific concepts
  - 그래서 COCO가 상위 Level로 묶이게 되고, ImageNet이 하위 Level로 묶이는 현상이 나타난다.

## Joint classification and detection
- COCO, ImageNet Top 9000개의 범주를 포함한 dataset을 WordTree를 이용해 결합시킨다.
- 이를 통해 9418개의 범주를 갖는 dataset을 사용한다.
- ImageNet과 COCO의 데이터 양을 고려해서 학습 시 비율을 4:1이 되게 조정한다.
- 학습 시, detection data의 경우 full loss를 역전파하고, classification data의 경우 classification loss 부분만 역전파한다.
- classification에서 label의 **하위 범주들은 학습에 고려하지 않고 상위 범주들만 고려**한다.
  - "dog" 범주인 경우 상위 범주인 "animal"은 고려해서 학습하지만 하위 범주인 "terrier"은 고려하지 않는다.
- YOLOv2는 9000개 이상의 범주를 detection할수 있는 YOLO9000이 된다.
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
- feature map 에서 모든 location에서의 offset들을 predict했다.
  - 좌표를 predict하는 것보다는 offset을 predict 하는 것이 netowrk 학습관점에서 문제를 더 단순화 할 수 있고 쉽게 학습한다고 한다.
- YOLOv2는 모든 FC-layer를 제거한다.
- bounding box predicting을 위해 anchor boxe를 사용한다.
- 하나의 Pooling layer 제거
  - Convolitional layer의 출력의 해상도를 높였다.
- 입력 이미지 크기를 448 에서 416으로 변경
  - feature map에서 location 수가 홀수가 나오기를 원했기 때문
  - 448x448에서 downsample 시 -> 14x14, 416x416 -> 13x13 feature map이 얻어진다. (홀수면 Single center가 생긴다.)
  - 보통 물체는 (특히 큰 Object)는 이미지의 중앙에 있는 경우가 많다. output feature map이 홀 수 인 것이 좋다.
  - Why? 짝수로 설정하면 중앙에 4개의 grid cell이 인접한다.
- anchor box를 이용하면 mAP는 감소하고 Recall이 증가한다. (개선의 여지는 있음)
- anchor box 사용하지 않았을 때 69.5 mAP와 81% recall
- anchor box 사용할 때 69.2 mAP와 88% recall을 얻었다.

## Dimension Clusters
- Anchor box를 수동으로 대략적으로 설정해서 사용하는데, 이는 비효율적이다.
  - 만약 사전에 더 좋은 anchor box 형태로 조정하면 학습이 더 잘이뤄질 수 있다.
- K-means clustering을 통해 training set에 대한 bounding box들을 사전에 좋은 것들을 찾는다.

![image](https://user-images.githubusercontent.com/69780812/141072276-6823fcc7-09fb-4f82-a9dd-497012d65381.png)
- k-means의 Euclidean distance를 사용한다.
- standard의 k-means(with 유클리디안)은 큰 박스들에 대해서 작은 박스들보다 더 많은 에러가 생긴다.
  - YOLOv2 팀은 좋은 IOU score로 이끌어줄 anchor 박스가 필요했다.
  - 이는 Box의 Size와는 무관하게 독립적인 것이다.

![image](https://user-images.githubusercontent.com/69780812/141073102-97fb8174-8d5a-494e-b24a-c7eed685fa35.png)
- 따라서 distance method를 위와 같은 것으로 정의하게 됐다.
- 그 이전 그림을 봤을 때, 높은 k일 수록 높은 정확도를 얻을 수 있지만 속도가 느려지므로 적절한 k=5값을 선택하게 된다.

![image](https://user-images.githubusercontent.com/69780812/141073483-d551dda4-1fc3-40ef-9a49-ffc1347cc247.png)
- hand-picked, 수동으로 대략 선정했던 것들보다 clustering을 했을 때 더 좋은 결과를 가져온다는 것을 확인할 수 있다.
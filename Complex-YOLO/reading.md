# Abstract
- Real-time 3D object detection 모델이다.
- multi-class 3D 박스들을 Cartesian space라고 하는 공간에서 추정하는 방식의 복잡한 regression 전략을 통해 YOLOv2를 확장한 Network를 설명한다.
- E-RPN을 제안한다. (Euler Region Proposal Network)
  - regression network에 허수, 실수부를 추가하여 object의 pose를 추정한다.
  - 이러한 기법이 하나의 angle estimation을 하도록 해준다. 즉, 복잡한 sapce, singularity들을 피하게 해준다.
  - Training 동안 일반화를 하는데 있어서 도움이 된다고 한다.

# Introduction
- Point Cloud Proecessing은 자율주행에서 중요하다.
  - Lidar 센서가 최근 엄청난 발전을 이루고 있다.
- 주변 환경의 3D Points를 실시간으로 주는 것은 객체를 감싸고있는 거리에대한 직접적인 측정을 할 수 있다는 장점이 갖게 해준다.
- 또한, 이러한 점이 postion, 3D에서 객체들의 향하고 있는 방향을 추정하는 자율주행에대한 Object Detection 알고리즘 개발을 할 수 있게해준다.
- Lidar point cloud는 이미지와 비교해서 sparse 하다는 특징이 있다. 게다가 순서도 없다.
  - Sparse & Unordered
- 2D Obejct Detection에서 주로 accuracy와 efficiency사이의 traide-off에 초점을 두고 연구가 진행중이다. 자율주행 관점에서 efficiency가 더 중요하다.
  - 그래서 RPN(region proposal netwokr)이나 grid 기반의 RPN-approach을 사용하는 것이다.
  - RPN과 같은 네트워크들은 임베디드 장치에서 돌아갈만큼 매우 효율적이고 정확하다.
- 딥러닝에서 Point Cloud 기반 3D bbox 추정방법은 3가지이다.
  - 1. Direct point cloud processing Using Multi-Layer-Perceptrons
  - 2. Translation of Point-Clouds into voxels or image stacks by using Convolutional Nerual Networks(CNN)
  - 3. Combined fusion approaches
    - 이건 뭔말인지 모르겠다. 참조논문 제목을 보니 자율주행분야의 기술이 합쳐진 것을 의미하는 것 같다.

## 1.1 Related Work

## 1.2 Contribution

# 2. Complex-YOLO
- point cloud에 대한 pre-processing 기반 grid를 설명한다.

## 2.1 Point Cloud Preprocessing
- single frame에서 3D point cloud들은 birds-eye-view RGB-map으로 변환된다.
  - 여기서는 벨로다인 HDL64 모델을 사용했다고 한다.
- RGB-map은 height, intensity, density로 인코딩딘다.
- grid map의 size는 아래와 같이 정의된다.
  - n : 1024
  - m : 512
  - 3D point cloud가 2D grid로 projection 될 수 있다.
  - g : about 8cm (? 이건 뭔지 모르겠다.)
- 효율성을 위해 multiple height map 대신 하나의 height map만 사용하고있다.
- 결과적으로 3개의 feature channel들이 전체 Point cloud에 대해 계산된다.
  - feature channel : z_r, z_g, z_b

![image](https://user-images.githubusercontent.com/69780812/144954094-475e813f-f128-4c4d-8216-d064f5912ae5.png)
- 세 개의 feature channel들이 convering area(Ohm) 영역내의 Point Cloud 전체에 대해 계산된다.

![image](https://user-images.githubusercontent.com/69780812/144954497-ebf4c7ea-353a-4a71-a345-217fb304bee4.png)
- 각 Point_i에 대해 grid로 mapping 하는 함수를 정의한다.
- 각 point_i을 특정 grid cell S_j(RGB-map)으로 mapping 하는 것이다.

![image](https://user-images.githubusercontent.com/69780812/144954647-579a215e-e349-4806-8822-3e2fa7aac9db.png)
- 각 픽셀에 대한 채널을 계산할 수 있게 된다.
- I : 벨로다인 intensity를 고려하는 것이다.
- N : P_ohm_i에서 S_j로 매핑되는 Point들의 수
- g : grid cell size에 대한 parameter
  - z_g : 최대 height로 encode 된다.
  - z_b : 최대 intensity
  - z_r : S_j로 매핑된 모든 점들에 대한 normalize된 density

## 2.2 Architecture
- Complex-YOLO는 bird-eye-view RGB-map을 사용한다.
- 그리고 단순화된 YOLOv2 CNN을 사용한다.
  - CNN은 YOLOv4든 바꿀 수 있을 것으로 보인다.
  - E-RPN과 complex angle regression으로 더 확장되어진 형태라고한다.

### Euler-Region-Proposal
- E-RPN은 3D position, object dimensions(b_w, b_h), probability(p), oritentation(b_pi)를 parse 한다.
  - probability : p0, p1, ... pn (class scores)
  - b_pi : feature map 으로부터 얻어낸다.

![image](https://user-images.githubusercontent.com/69780812/144963087-af647ef5-af20-4374-964d-c47e9e336bc0.png)
- 적절한 orientation으 얻기위해서 Grid_RPN approach를 사용한다.

![image](https://user-images.githubusercontent.com/69780812/144963141-e1ea5e5a-344f-43ec-bf41-7a4f3ebb469a.png)
- 위의 arg는 complex angle이다.
  - 이를 추가해서 Grid-RPN approach를 사용해 bbox 정의를 수정할 수 있었다.
- 이와 같은 extension으로 인해 E-RPN은 정확한 object orientation들을 추정한다.
  - 직접적으로 네트워크에 임베드 되어진 실수부, 허수부를 기반으로 추정한다.
- 각 grid cell에 대해서 5개의 Object를 predict한다.
  - probability score, class score를 포함한다.
  - 각 grid cell당 75개의 feature가 산출된다.

![image](https://user-images.githubusercontent.com/69780812/144963830-72b13f4f-79a5-4213-9ac3-90bb5cb1725d.png)
- Complex-YOLO의 Design이다.

### Anchor Box Desing
- grid cell당 5개의 box를 prior로 둔다.
  - training 동안 더 잘 수렴하도록 도와준다.
- angle regression 때문에 자유도 즉, 가능한 prior의 수는 증가했지만 효율성을 위해서 prediction의 수를 증가시키지는 않았다.
  - 세 개의 다른 size, 두 개의 angle direction을 prior로 정의했다.
  - Kitti dataset 내부 box들의 분산을 기반으로 설정했다.
    - 1. vehicle size (heading up)
    - 2. vehicle size (heading down)
    - 3. cyclist size (heading up)
    - 4. cyclist size (heading down)
    - 5. pedestrian size (heading left)

### Complex Angle Regression
- 각 Object에 대한 orientation angle은 t_im, t_re 라고하는 parameter들을 regression해서 계산할 수 있다.
  - 
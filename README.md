# Shot

Pytorch implement
of [[ICML-2020] Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](http://proceedings.mlr.press/v119/liang20a.html)

### Experiment Details

| 범주     | 카테고리                | 내용                                                         |
| -------- | ----------------------- | ------------------------------------------------------------ |
| 데이터   | Train Augmentation      | Resize, RandomCrop, RandomHorizontalFlip, Normalize          |
|          | Test Augmentation       | Resize, CenterCrop, Normalize                                |
| 모델     | 구조                    | ResNet50 (Pre-trained on MS1V3)                              |
| 학습도구 | Optimizer               | SGD, lr=0.01, weight_decay=1e-3, momentum=0.9                |
|          | Criterion(source train) | CrossEntropyLoss With LabelSmooth(0.1)                       |
|          | Criterion(target train) | CrossEntropyLoss, Entropy Loss, IM Loss                      |
|          | LR Scheduler            | 1 + gamma * iter_num / max_iter) ** power 만큼 epoch마다 감소 |
| 학습     | Epoch(source/target)    | 50 / 15                                                      |
|          | Batch size              | 128                                                          |
| 평가     | Softmax                 | CDA, PDA                                                     |
|          | Softmax & threshold     | 라벨에 해당하는 이미지는 softmax, 라벨에 해당하지 않는 이미지는 1,0 |

### Result

#### Office-Home

| PDA                      | Ar->Cl | Ar->Pr | Ar->Re |
| ------------------------ | :----: | :----: | :----: |
| source only (repository) |  46.4  |  71.8  |  80.6  |
| source only (paper)      |  45.2  |  70.4  |  81.0  |
| SHOT (repository)        |  61.0  |  82.8  |        |
| SHOT (paper)             |  64.8  |  85.2  |  92.7  |

#### RMFD

*Dataset Description*

|            | # Image | # class |
| ---------- | ------- | ------- |
| Non Masked | 87228   | 442     |
| Masked     | 1945    | 442     |

*Result*

| CDA                                      | Unmask -> Mask  |
| ---------------------------------------- | --------------- |
| source only(unmask)                      | 9.05            |
| SHOT                                     | 0.5 (학습 안됌) |
| arcfaceBackbone transfer learning unmask | 22.7            |
| arcfaceBackbone + SHOT                   | 0.3 (학습 안됌) |

Pseudo labeling이 제대로 되지 않음. 이전보다 정확도가 더 떨어짐. class의 수(443)가 너무 많아서 제대로 학습되지 않는 것으로 보임
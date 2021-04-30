모든 조건은 같고, transform만 다르게 하여 실험하였다.

Model : segmentationmodels.pytorch efficientnet-b0 UNet

epoch : 8

batch_size : 8

image_size : 256

요약
1. Rotate보다 RandomRotate90이 더 낫다.
2. MotionBlur, OpticalDistortion은 있는 게 더 낫다.
3. ElasticDistortion, GridDropout은 없는 게 더 낫다.
4. GaussNoise를 사용하면 학습을 거의 못한다.

## 1.

    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0)
    ], p=2/3),
    ToTensorV2()
    
    Validation #1 mIoU: 0.1659
    Validation #2 mIoU: 0.1735
    Validation #3 mIoU: 0.1945
    Validation #4 mIoU: 0.2063
    Validation #5 mIoU: 0.2393
    Validation #6 mIoU: 0.2630
    Validation #7 mIoU: 0.2792
    Validation #8 mIoU: 0.2853

## 2.
    OneOf transform을 없앴더니 validation mIoU가 떨어졌다.
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    ToTensorV2()
    
    Validation #1 mIoU: 0.1566
    Validation #2 mIoU: 0.1707
    Validation #3 mIoU: 0.2066
    Validation #4 mIoU: 0.2106
    Validation #5 mIoU: 0.2366
    Validation #6 mIoU: 0.2490
    Validation #7 mIoU: 0.2615
    Validation #8 mIoU: 0.2622

## 3.
    A.Resize(256, 256),
    A.ElasticTransform(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0)
    ], p=2/3),
    ToTensorV2()
    Validation #1 mIoU: 0.1633
    Validation #2 mIoU: 0.1716
    Validation #3 mIoU: 0.1835
    Validation #4 mIoU: 0.1903
    Validation #5 mIoU: 0.2288
    Validation #6 mIoU: 0.2339
    Validation #7 mIoU: 0.2633
    Validation #8 mIoU: 0.2709

## 4.

    ElasticTransform은 어디다 넣어도 성능이 떨어지는 것 같다.
    
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0)
    ], p=0.75),
    ToTensorV2()
    
    Validation #1 mIoU: 0.1648
    Validation #2 mIoU: 0.1722
    Validation #3 mIoU: 0.1885
    Validation #4 mIoU: 0.1939
    Validation #5 mIoU: 0.2310
    Validation #6 mIoU: 0.2506
    Validation #7 mIoU: 0.2736
    Validation #8 mIoU: 0.2793
    
## 5.
    
    GaussNoise를 넣으니 학습이 거의 안 되었다.
    train_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.GaussNoise((0.04, 0.2), p=1.0),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0)
    ], p=2/3),
    ToTensorV2()
    ])
    Validation #1 mIoU: 0.0703
    Validation #2 mIoU: 0.0603
    Validation #3 mIoU: 0.0597
    Validation #4 mIoU: 0.0602
    Validation #5 mIoU: 0.0615
    Validation #6 mIoU: 0.0608
    Validation #7 mIoU: 0.0612
    Validation #8 mIoU: 0.0592
    
## 6.

    grid dropout을 쓰니 성능이 안좋아졌다.
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0)
    ], p=2/3),
    A.GridDropout(p=1.0),
    ToTensorV2()
    
    Validation #1 mIoU: 0.1565
    Validation #2 mIoU: 0.1634
    Validation #3 mIoU: 0.1719
    Validation #4 mIoU: 0.1821
    Validation #5 mIoU: 0.2018
    Validation #6 mIoU: 0.2149
    Validation #7 mIoU: 0.2224
    Validation #8 mIoU: 0.2393
 
 ## 7.
 
    dropout을 조금만 쓰니 안 쓴 것 대비 조금만 떨어졌다.
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
    ),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.OpticalDistortion(p=1.0),
        A.GridDropout(p=1.0)
    ], p=0.75),
    ToTensorV2()
    Validation #1 mIoU: 0.1640
    Validation #2 mIoU: 0.1717
    Validation #3 mIoU: 0.1869
    Validation #4 mIoU: 0.1950
    Validation #5 mIoU: 0.2215
    Validation #6 mIoU: 0.2530
    Validation #7 mIoU: 0.2676
    Validation #8 mIoU: 0.2778

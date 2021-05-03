import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
    Augmentation 설정
"""

# 항상 고정으로 사용하는 augmentation
train_transform = A.Compose([
                             A.OneOf([A.HorizontalFlip(p=1),
                                      A.VerticalFlip(p=1),
                                      A.RandomRotate90(p=1)], p=1),
                             A.OneOf([A.MotionBlur(p=1),
                                      A.GaussianBlur(p=1),
                                      A.OpticalDistortion(p=1)], p=1),
                             A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2()
                          ])

val_transform = A.Compose([
                           A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2()
                          ])

test_transform = A.Compose([
                            A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),    
                            ToTensorV2()
                            ])


# ElasticTranspose 추가 0.6006
train_transform = A.Compose([
                             A.OneOf([A.HorizontalFlip(p=1),
                                      A.VerticalFlip(p=1),
                                      A.RandomRotate90(p=1)], p=1),
                             A.OneOf([A.MotionBlur(p=1),
                                      A.GaussianBlur(p=1),
                                      A.OpticalDistortion(p=1)], p=1),
                             A.ElasticTransform(alpha_affine=100, interpolation=2,p=1.0),
                             A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2()
                          ])


# GridDropOut 추가 0.5820
train_transform = A.Compose([
                             A.OneOf([A.HorizontalFlip(p=1),
                                      A.VerticalFlip(p=1),
                                      A.RandomRotate90(p=1)], p=1),
                             A.OneOf([A.MotionBlur(p=1),
                                      A.GaussianBlur(p=1),
                                      A.OpticalDistortion(p=1)], p=1),
                             A.GridDropout(ratio=0.4, p=1),
                             A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2()
                          ])


# ElasticTransform, GlassBlur 추가 0.5798
train_transform = A.Compose([
                             A.OneOf([A.HorizontalFlip(p=1),
                                      A.VerticalFlip(p=1),
                                      A.RandomRotate90(p=1)], p=1),
                             A.OneOf([A.MotionBlur(p=1),
                                      A.GaussianBlur(p=1),
                                      A.OpticalDistortion(p=1),
                                      A.GlassBlur(p=1)], p=1),
                             A.ElasticTransform(alpha_affine=100, interpolation=2,p=1.0),
                             A.Resize(256, 256),
                             A.Normalize(
                                mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
                            ),                           
                            ToTensorV2()
                          ])
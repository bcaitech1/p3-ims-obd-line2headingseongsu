import segmentation_models_pytorch as smp

batch_size = 16   # Mini-batch size
num_epochs = 100
learning_rate = 1e-4

ENCODER = "resnext50_32x4d"
ENCODER_WEIGHTS = 'imagenet'

model = smp.DeepLabV3Plus(encoder_name=ENCODER,
                          encoder_weights=ENCODER_WEIGHTS,
                          classes=12)

model = model.to(device)
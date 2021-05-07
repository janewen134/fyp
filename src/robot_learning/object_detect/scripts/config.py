
from easydict import EasyDict
Cfg = EasyDict()

Cfg.batch = 256
Cfg.subdivisions = 32
Cfg.width = 512 
Cfg.height = 512 
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.jitter = 0.3
Cfg.mosaic = True

Cfg.learning_rate = 0.000001
Cfg.burn_in = 250
Cfg.max_batches = 125125 
Cfg.steps = [100000, 120000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.use_attention = True

Cfg.classes = 20
#80
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height

Cfg.cosine_lr = True
#False
Cfg.smoooth_label = False
Cfg.TRAIN_OPTIMIZER = 'adam'
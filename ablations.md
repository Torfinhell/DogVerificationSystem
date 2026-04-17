# Ablations for Barkopedia + ECAPA-TDNN

## Common Configuration
```bash
COMMON="writer.logger.mode=online sample_rate=8000 trainer.compile.enabled=true trainer.compile.call._target_=torch.compile trainer.compile.call.mode=default trainer.compile.call.fullgraph=false trainer.compile.call.dynamic=false trainer.compile.call.backend=cudagraphs trainer.amp.enabled=false -cn=x_vector"
```

---

## 1) Ablation #1: Loss Function Parameters (margin `m`, scale `s`)

### m=0.1, s=30
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.1 loss_function.scale=30 trainer.save_dir=saved/ablation/loss_m0.1_s30 writer.logger.run_name="loss_m0.1_s30" $COMMON
```

### m=0.1, s=40
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.1 loss_function.scale=40 trainer.save_dir=saved/ablation/loss_m0.1_s40 writer.logger.run_name="loss_m0.1_s40" $COMMON
```

### m=0.1, s=50
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.1 loss_function.scale=50 trainer.save_dir=saved/ablation/loss_m0.1_s50 writer.logger.run_name="loss_m0.1_s50" $COMMON
```

### m=0.2, s=30
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.2 loss_function.scale=30 trainer.save_dir=saved/ablation/loss_m0.2_s30 writer.logger.run_name="loss_m0.2_s30" $COMMON
```

### m=0.2, s=40
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.2 loss_function.scale=40 trainer.save_dir=saved/ablation/loss_m0.2_s40 writer.logger.run_name="loss_m0.2_s40" $COMMON
```

### m=0.2, s=50
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.2 loss_function.scale=50 trainer.save_dir=saved/ablation/loss_m0.2_s50 writer.logger.run_name="loss_m0.2_s50" $COMMON
```

### m=0.3, s=30
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.3 loss_function.scale=30 trainer.save_dir=saved/ablation/loss_m0.3_s30 writer.logger.run_name="loss_m0.3_s30" $COMMON
```

### m=0.3, s=50
```bash
MPLBACKEND=Agg uv run -- python train.py loss_function.margin=0.3 loss_function.scale=50 trainer.save_dir=saved/ablation/loss_m0.3_s50 writer.logger.run_name="loss_m0.3_s50" $COMMON
```

---

## 2) Ablation #2: Batch Sampler On/Off

### Batch Sampler On
```bash
MPLBACKEND=Agg uv run -- python train.py batch_sampler=hpm trainer.save_dir=saved/ablation/batch_sampler_on writer.logger.run_name="batch_sampler_on" $COMMON
```

### Batch Sampler Off
```bash
MPLBACKEND=Agg uv run -- python train.py ~batch_sampler trainer.save_dir=saved/ablation/batch_sampler_off writer.logger.run_name="batch_sampler_off" $COMMON
```

---

## 3) Ablation #3: Backend PLDA On
```bash
MPLBACKEND=Agg uv run -- python train.py backends=plda trainer.save_dir=saved/ablation/plda_on writer.logger.run_name="plda_on" $COMMON
```

---

## 4) Ablation #4: Augmentation Types

### No Augmentation
```bash
MPLBACKEND=Agg uv run -- python train.py transforms.instance_transforms=basic trainer.save_dir=saved/ablation/aug_none writer.logger.run_name="aug_none" $COMMON
```

### MFCC2 Augmentation (Musan + RIR)
```bash
MPLBACKEND=Agg uv run -- python train.py transforms.instance_transforms=mfcc_2 trainer.save_dir=saved/ablation/aug_mfcc2 writer.logger.run_name="aug_mfcc2" $COMMON
```

### Full Augmentation
```bash
MPLBACKEND=Agg uv run -- python train.py transforms.instance_transforms=augmented trainer.save_dir=saved/ablation/aug_augmented writer.logger.run_name="aug_augmented" $COMMON
```

---

## 5) Final Run: ECAPA-TDNN with Best Hyperparameters + AMP
```bash
MPLBACKEND=Agg uv run -- python train.py --config-name ecappa_tdnn \
  sample_rate=8000 \
  batch_sampler=hpm \
  backends=plda \
  loss_function.margin=0.2 \
  loss_function.scale=40 \
  transforms.instance_transforms=augmented \
  trainer.compile.enabled=true \
  trainer.compile.call._target_=torch.compile \
  trainer.compile.call.mode=default \
  trainer.compile.call.fullgraph=false \
  trainer.compile.call.dynamic=false \
  trainer.compile.call.backend=inductor \
  trainer.amp.enabled=true \
  trainer.amp.dtype=float16 \
  trainer.save_dir=saved/best_ecappa_tdnn \
  writer.logger.run_name="best_ecappa_tdnn"
```
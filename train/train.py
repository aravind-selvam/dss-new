import sys
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    TQDMProgressBar
)
import numpy as np
from tqdm import tqdm
#from pytorch_lightning.loggers import WandbLogger

from datamodule.datamodule import SleepDataModule
from model.model import PLSleepModel
from common.metric import event_detection_ap
from common.postprocess import post_process_for_seg

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
    
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

def train(cfg, env):
    seed_everything(cfg["seed"])

    # init lightning model
    datamodule = SleepDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    model = PLSleepModel(cfg, datamodule.valid_event_df, len(cfg["features"]), len(cfg["labels"]), cfg["duration"])

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg["trainer"]["monitor"],
        mode=cfg["trainer"]["monitor_mode"],
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    if env == "colab":
        progress_bar = MyProgressBar()
    else:
        progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    #pl_logger = WandbLogger(
    #    name=cfg["exp_name"],
    #    project="child-mind-institute-detect-sleep-states",
    #)
    #pl_logger.log_hyperparams(cfg)

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg["trainer"]["num_gpus"],
        accelerator=cfg["trainer"]["accelerator"],
        precision=16 if cfg["trainer"]["use_amp"] else 32,
        # training
        fast_dev_run=cfg["trainer"]["debug"],  # run only 1 train batch and 1 val batch
        max_epochs=cfg["trainer"]["epochs"],
        max_steps=cfg["trainer"]["epochs"] * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        #logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
    )

    trainer.fit(model, datamodule=datamodule)

    # load best weights
    model = PLSleepModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg["features"]),
        num_classes=len(cfg["labels"]),
        duration=cfg["duration"],
    )
    weights_path = cfg["dir"]["model_path"]  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)


def do_train(model, dataloader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        inputs, labels = batch['feature'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        do_mixup = np.random.rand() < cfg["aug"]["mixup_prob"]
        do_cutmix = np.random.rand() < cfg["aug"]["cutmix_prob"]
        outputs = model(inputs, labels, do_mixup, do_cutmix)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    # Calculate the average loss
    average_loss = total_loss / len(dataloader)
    print("Training loss", average_loss)

    return average_loss

def do_validate(model, dataloader, device, cfg):
    model.eval()
    total_loss = 0
    validation_step_outputs = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels = batch['feature'].to(device), batch['label'].to(device)
            
            # Predict the outputs using the duration parameter
            outputs = model.model.predict(inputs, cfg["duration"], labels)

            # Detach and move the loss to CPU
            loss_value = outputs.loss.detach().item()
            total_loss += loss_value

            # Optionally log or print loss here

            # Append to validation_step_outputs
            validation_step_outputs.append(
                (
                    batch.get("key", None),  # Assuming 'key' is optional
                    outputs.labels.detach().cpu().numpy(),
                    outputs.preds.detach().cpu().numpy(),
                    loss_value,
                )
            )

    # Calculate the average loss
    average_loss = total_loss / len(dataloader)
    print("Validation loss", average_loss)

    return average_loss, validation_step_outputs

def train2(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 모듈 초기화
    datamodule = SleepDataModule(cfg)

    # 모델 초기화
    model = PLSleepModel(cfg, datamodule.valid_event_df, len(cfg["features"]), len(cfg["labels"]), cfg["duration"]).to(device)

    # 옵티마이저 및 손실 함수 설정
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg["optimizer"]["lr"])
    best_val_score = float('inf')
    for epoch in tqdm(range(cfg["trainer"]['epochs'])):
        train_loss = do_train(model, datamodule.train_dataloader(), optimizer, device, cfg)
        val_loss, validation_step_outputs = do_validate(model, datamodule.val_dataloader(), device, cfg)

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        keys = []
        for x in validation_step_outputs:
            keys.extend(x[0])
        labels = np.concatenate([x[1] for x in validation_step_outputs])
        preds = np.concatenate([x[2] for x in validation_step_outputs])
        losses = np.array([x[3] for x in validation_step_outputs])
        loss = losses.mean()

        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds,
            score_th=cfg["pp"]["score_th"],
            distance=cfg["pp"]["distance"],
        )
        score = event_detection_ap(model.val_event_df.to_pandas(), val_pred_df.to_pandas())
        print("val_score: ", score)

        # 검증 손실이 개선되었는지 확인하고 모델 저장
        if score < best_val_score:
            best_val_score = score
            print("save model with best_val_score: ", score)
            weights_path = cfg["dir"]["model_path"]
            torch.save(model.model.state_dict(), weights_path)

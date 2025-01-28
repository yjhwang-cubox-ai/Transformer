import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformer import LitTransformerModule
from datamodule import WMT14DataModule

import wandb


def main():
    project_name = 'transformer_train'
    run_name = '241006_1st'

    # Logger setting
    wandb.finish()
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    wandb_logger = WandbLogger(project=project_name, name=run_name, config={}, log_model=False)    
    print("login!")

    # 데이터 모듈 초기화
    data_module = WMT14DataModule()

    # 모델 초기화
    model = LitTransformerModule()
    
    # Training
    epochs = 30
    num_gpus = 1
    num_nodes = 1

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=num_gpus,
        num_nodes=num_nodes,
        val_check_interval=0.1,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='./checkpoints',
                filename='transformer-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min'
            )
        ]
    )

    # 학습 시작
    trainer.fit(model, data_module)
    # 테스트
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()
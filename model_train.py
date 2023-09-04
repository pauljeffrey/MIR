# my_script.py

from memory_profiler import profile
from new_train.py import *


@profile
def train(cfg: DictConfig):
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=cfg.training.gradient_accumulation_steps, gradient_clipping=1.0)
    accelerator = Accelerator(  deepspeed_plugin =deepspeed_plugin) #,  mixed_precision='fp16',
    
    accelerator.wait_for_everyone()
    device= accelerator.device
    

    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
    
    if not cfg.model.from_checkpoint:
        model = load_model(cfg)
        epoch = None
        loss = None
        # Optimizer
        # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
        optimizer_cls = (
            torch.optim.AdamW #Adafactor #
            # if accelerator.state.deepspeed_plugin is None
            # or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            # else DummyOptim
        )
        optimizer = optimizer_cls(model.parameters(), lr=cfg.training.learning_rate)


        # if (
        #     accelerator.state.deepspeed_plugin is None
        #     or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        # ):
    else:
        model, optimizer, epoch, loss = load_model(cfg)
        
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )
    # else:
    #     lr_scheduler = DummyScheduler(
    #         optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps
    #     )

    transform = transforms.Compose(
    [
        transforms.RandomVerticalFlip(0.45),
        transforms.RandomHorizontalFlip(0.45),
        transforms.RandomCrop((224, 224), padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = MyDataset(train_dir, transform=transform)
    eval_dataset = MyDataset(eval_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    for epoch in range(epoch + 1, cfg.training.num_epochs + 1):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = custom_loss(outputs, labels)
            loss.backward()
            
            if i % cfg.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=-1) == labels).sum().item()

            del images, labels, loss, outputs
            torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = custom_loss(outputs, labels)

                eval_loss += loss.item()
                eval_acc += (outputs.argmax(dim=-1) == labels).sum().item()

                del images, labels, loss, outputs
                torch.cuda.empty_cache()

            eval_loss /= len(eval_loader)
            eval_acc /= len(eval_loader.dataset)

            logger.info(
                f"Epoch [{epoch}/{cfg.training.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
            )

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_model(model, optimizer, epoch, loss)

    return best_eval_acc


if __name__ == "__main__":
    cfg = OmegaConf.load("/kaggle/working/MIR/conf/config.yaml") #
    train(cfg)
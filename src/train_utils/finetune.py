import os
import torch
import logging
import numpy as np
from tqdm import tqdm

# train utils
from train_utils.eval_functions import val_and_logging
from train_utils.optimizer import define_optimizer
from train_utils.lr_scheduler import define_lr_scheduler

# utils
from general_utils.time_utils import time_sync
from general_utils.weight_utils import load_model_weight, set_learnable_params_finetune
from params.output_paths import set_finetune_weights


def finetune(
    args,
    classifier,
    augmenter,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    classifier_loss_func,
    num_batches,
):
    """Fine tune the backbone network with only the class layer."""
    # Load the pretrained feature extractor -- unless this is the
    # random-init control condition, in which case we deliberately skip
    # this and keep the freshly-initialized (never trained) backbone.
    if args.random_init_backbone:
        logging.info("=\t[Control condition] Using randomly initialized backbone -- pretrained weights NOT loaded")
    else:
        pretrain_weight = os.path.join(args.weight_folder, f"{args.dataset}_{args.model}_pretrain_latest.pt")
        classifier = load_model_weight(args, classifier, pretrain_weight, load_class_layer=False)
    learnable_parameters = set_learnable_params_finetune(args, classifier)

    # Init the optimizer, scheduler, and weight files
    optimizer = define_optimizer(args, learnable_parameters)
    lr_scheduler = define_lr_scheduler(args, optimizer)
    best_weight, latest_weight = set_finetune_weights(args)

    # Standardize regression targets for training stability -- raw wheel
    # speed has a large scale (hundreds), which makes MSE loss huge and
    # gradient descent poorly conditioned for a shallow probe head. Stats
    # are computed from the TRAIN split only (per LOAO fold), never
    # val/test, to avoid leakage across the held-out animal/sessions.
    if args.dataset.startswith("Shikano"):
        all_train_labels = []
        for _, labels in train_dataloader:
            all_train_labels.append(labels)
        all_train_labels = torch.cat(all_train_labels)
        args.label_mean = all_train_labels.mean().item()
        args.label_std = all_train_labels.std().item()
        logging.info(f"=\t[Label standardization] mean={args.label_mean:.3f}, std={args.label_std:.3f}")

    # Training loop
    logging.info("---------------------------Start Fine Tuning-------------------------------")
    start = time_sync()
    # NOTE: was `best_val_acc = 0`. R^2 can be negative early in training
    # (worse than predicting the mean), which meant no checkpoint would
    # ever be saved as "best" until R^2 turned positive. -inf ensures the
    # first eval always establishes a baseline "best" checkpoint.
    best_val_acc = -np.inf

    val_epochs = 5
    for epoch in range(args.dataset_config[args.learn_framework]["finetune_lr_scheduler"]["train_epochs"]):
        if epoch > 0:
            logging.info("-" * 40 + f"Epoch {epoch}" + "-" * 40)

        # set model to train mode
        classifier.train()

        # training loop
        train_loss_list = []
        for i, (time_loc_inputs, labels) in tqdm(enumerate(train_dataloader), total=num_batches):
            # move to target device, FFT, and augmentations
            aug_freq_loc_inputs, labels = augmenter.forward("no", time_loc_inputs, labels)

            # forward pass
            logits = classifier(aug_freq_loc_inputs)
            if args.dataset.startswith("Shikano"):
                logits = logits.squeeze(-1)  # (batch, 1) -> (batch,), regression output
                norm_labels = (labels - args.label_mean) / args.label_std
                loss = classifier_loss_func(logits, norm_labels)
            else:
                loss = classifier_loss_func(logits, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss_list.append(loss.item())

        # validation and logging
        if epoch % val_epochs == 0:
            train_loss = np.mean(train_loss_list)
            val_metric, val_loss = val_and_logging(
                args,
                epoch,
                classifier,
                augmenter,
                val_dataloader,
                test_dataloader,
                classifier_loss_func,
                train_loss,
            )

            # Save the latest model
            torch.save(classifier.state_dict(), latest_weight)

            # Save the best model according to validation result
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                torch.save(classifier.state_dict(), best_weight)

        # Update the learning rate scheduler
        lr_scheduler.step(epoch)

    end = time_sync()
    logging.info("------------------------------------------------------------------------")
    logging.info(f"Total processing time: {(end - start): .3f} s")
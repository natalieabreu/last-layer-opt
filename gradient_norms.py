import os

print(os.environ['PYTHONPATH'], flush=True)


import argparse

import copy
import numpy as np
import wandb
from functools import partial

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from resnet import CustomResNetPartial, MyLayerNorm, MyGroupNorm
from cifar5m import CIFAR5MDataset
from utils import load_config, generate_sweep_config

# print pythonpath

CIFAR5M_DIR = '/n/netscratch/kempner_barak_lab/Everyone/cifar5m'

# Evaluation function
def evaluate_model(model, dataloader, loss_fn, eval_batches=1):
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= eval_batches: break
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            accuracies.append((outputs.argmax(1) == labels).float().mean().item())
    model.train()
    # print(losses)
   
    return np.mean(losses), np.mean(accuracies)

# Main training function
def train_model(model, trainloader, testloader, config):
    # Set hyperparameters

    # read params from config
    LR = config['LR']
    LLLR = config['LAST_LAYER_LR']
    INNER_LR = config['INNER_OPT_LR']
    WARMUP = config['WARMUP']
    STEPS = config['STEPS']
    DATASET = config['DATASET']
    OPT = config['OPT']
    INNER_STEPS = config['INNER_STEPS']

    # Initialize wandb
    wandb.init(project=config['wandb_project'], entity='harvardml', config=config)
    wandb.define_metric("last_layer_step")
    wandb.define_metric("last_layer_loss", step_metric="last_layer_step")
    wandb.define_metric("last_layer_lr", step_metric="last_layer_step")
    wandb.define_metric("update_step")
    wandb.define_metric("loss", step_metric="update_step")
    wandb.define_metric("eval/loss", step_metric="update_step")
    wandb.define_metric("eval/accuracy", step_metric="update_step")
    wandb.define_metric("lr", step_metric="update_step")
    wandb.define_metric("lllr", step_metric="update_step")
    for name, param in model.named_parameters():
        wandb.define_metric(f"post_ll_grad_norm/{name}", step_metric="update_step")
        wandb.define_metric(f"main_grad_norm/{name}", step_metric="update_step")
        wandb.define_metric(f"grad_norm_ratio/{name}", step_metric="update_step")


    # Optimizers and Schedulers
    ll_param_list = ['fc.weight', 'fc.bias']
    param_names = [name for name, param in model.named_parameters()]
    for param in ll_param_list: # safeguard in case model is changed
        assert param in param_names, f"{param} not found in model.named_parameters()"
    ll_params = list(filter(lambda kv: kv[0] in ll_param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in ll_param_list, model.named_parameters()))

    opt = AdamW([
            {'params': [param[-1] for param in base_params]},
            {'params': [param[-1] for param in ll_params], 'lr': LLLR}
        ], lr=LR, betas=(0.9, 0.95), eps=1e-15)
    scheduler = get_constant_schedule_with_warmup(opt, WARMUP)

    
    loss_fn = CrossEntropyLoss()
    step = 1

    while step <= STEPS:
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()

         
            if OPT == 'STANDARD+GN':
                original_fc_state = copy.deepcopy(model.fc.state_dict())
                # opt_bias = AdamW(model.fc.parameters(), lr=LR3, betas=(0.95, 0.95), eps=1e-15)
                opt_ll = SGD(model.fc.parameters(), lr=INNER_LR, momentum=0.9)
        
                inner_scheduler = get_cosine_schedule_with_warmup(opt_ll, 0, INNER_STEPS)
                model.train()
                intermediate_outputs = model.forward2(images).detach()

                for j in range(INNER_STEPS):
                    last_layer_outputs = model.fc(intermediate_outputs)
    
                    loss = loss_fn(last_layer_outputs, labels)
    
                    opt_ll.zero_grad()
                    loss.backward()
                    
                    wandb.log({'last_layer_step': step*INNER_STEPS + j, 'last_layer_loss': loss.item(), 'last_layer_lr': inner_scheduler.get_last_lr()[0]})


                    opt_ll.step()
                    inner_scheduler.step()

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                opt.zero_grad()
                loss.backward()

                ll_grad_norms = {}
                # Compute and log gradient norms for all model parameters
                sorted_params = sorted(model.named_parameters(), key=lambda kv: kv[0])
                for name, param in sorted_params:
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()  # L2 norm
                        wandb.log({f"post_ll_grad_norm/{name}": grad_norm, 'update_step': step})
                        ll_grad_norms[name] = grad_norm
        
            
                model.fc.load_state_dict(original_fc_state)

            # Main model update
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            eval_loss, eval_acc = evaluate_model(model, testloader, loss_fn)
            print(loss, eval_loss)
            opt.zero_grad()
            loss.backward()

            ratios = {}
            # Compute and log gradient norms for all model parameters
            sorted_params = sorted(model.named_parameters(), key=lambda kv: kv[0])
            for name, param in sorted_params:
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()  # L2 norm
                    wandb.log({f"main_grad_norm/{name}": grad_norm, 'update_step': step})

                    if OPT == 'STANDARD+GN':
                        if name in ll_grad_norms:
                            ratios[name] = grad_norm / (ll_grad_norms[name] +  1e-20)
                            wandb.log({f"grad_norm_ratio/{name}": ratios[name], 'update_step': step})
                    
            opt.step()
            scheduler.step()
            
            # Logging
            wandb.log({
                'update_step': step,
                'loss': loss.item(),
                'eval/loss': eval_loss,
                'eval/accuracy': eval_acc,
                'lr': scheduler.get_last_lr()[0],
                'lllr': scheduler.get_last_lr()[1]
            })
     
            
            step += 1
            if step > STEPS:
                break

    # Final evaluation logging
    eval_loss, eval_acc = evaluate_model(model, testloader, loss_fn)
    wandb.log({'update_step': step, 'eval/loss': eval_loss})
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config

    config = load_config(config_path)
    slurm_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    config = generate_sweep_config(config, slurm_index)

    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Prepare datasets
    if config['DATASET'] == 'cifar5m':
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_dataset = CIFAR5MDataset(CIFAR5M_DIR, transform=transform, train=True)
        test_dataset = CIFAR5MDataset(CIFAR5M_DIR, transform=transform, train=False)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Loaders
    trainloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=10_000, shuffle=False, num_workers=4)

    # Initialize model and parameter combinations
    NORM = config.get('NORM', "BATCH_NORM")

    if config.get('NORM', "BATCH_NORM") == 'BATCH_NORM':
        norm = partial(nn.BatchNorm2d, momentum=1.0)
    elif NORM == 'LAYER_NORM':
        norm = MyLayerNorm

    # elif NORM == 'GROUP_NORM':
    #     norm = MyGroupNorm

    model = CustomResNetPartial(norm_layer=norm).cuda()

    # Run training
    train_model(model, trainloader, testloader, config)

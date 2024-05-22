# train.py
import time
import copy
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from dataset import Clip_Rescale, CTDataset
from model import QualitySentinel


with open('label_embedding.pkl', 'rb') as file:
    embedding_dict = pickle.load(file)


def weighted_mse_loss(input, labels):
    # weights for different intervals
    weights = torch.ones_like(labels)
    weights[labels <= 0.3] = 7
    weights[(labels > 0.3) & (labels <= 0.5)] = 5
    weights[(labels > 0.5) & (labels <= 0.7)] = 3
    weights[(labels > 0.7) & (labels <= 0.9)] = 2
    weights[(labels > 0.9) & (labels <= 1.0)] = 1
    
    loss = F.mse_loss(input, labels, reduction='none')
    weighted_loss = loss * weights

    return weighted_loss.mean()


def cosine_similarity_matrix(embeddings):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return torch.mm(embeddings, embeddings.t())


def find_pairs_with_hungarian(similarity_matrix):
    similarity_matrix = similarity_matrix.cpu().numpy()
    
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    cost_matrix = -similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def optimal_pair_ranking_loss(predictions, targets, embeddings):
    sim_matrix = cosine_similarity_matrix(embeddings)

    pairs = find_pairs_with_hungarian(sim_matrix)

    # paired ranking loss
    loss = 0
    for i, j in pairs:
        pred_diff = predictions[i] - predictions[j]
        target_diff = targets[i] - targets[j]
        loss += F.relu(-pred_diff * target_diff + 1e-4)

    return loss / len(pairs)


def main():
    # Hyperparameters
    model_name = 'resnet50'
    train_samples = 40
    epochs = 30
    batch_size = 128
    num_workers = 8
    learning_rate = 0.001
    weight_decay = 1e-4
    info_interval = 1
    eval_interval = 1
    TRAIN_DATA_PATH = 'Quality_Sentinel_data_50samples/train'
    VALID_DATA_PATH = 'Quality_Sentinel_data_50samples/val'
    MODEL_SAVE_PATH = 'best_resnet50_model_40_samples.pth'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    transform_ct = transforms.Compose([
        Clip_Rescale(min_val=-200, max_val=200),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print('Loading data...')
    train_dataset = CTDataset(TRAIN_DATA_PATH, transform_ct, transform_mask, mode='train', num_samples=train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    valid_dataset = CTDataset(VALID_DATA_PATH, transform_ct, transform_mask, mode='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, Criterion and Optimizer
    model = QualitySentinel(hidden_dim=50, backbone=model_name, embedding='text_embedding').to(device)
    model = model.to(device)
    
    criterion1 = weighted_mse_loss
    criterion2 = optimal_pair_ranking_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print('\nTraining Start!\n')
    start = time.time()
    max_val_coef = 0.0
    gt_dices = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        iter_start = time.time()
        for i, (ct, pred_mask, gt_mask, dice, mask_class) in enumerate(train_loader):
            ct, pred_mask, gt_mask, dice, mask_class = ct.to(device), pred_mask.to(device), gt_mask.to(device), dice.unsqueeze(1).to(device), mask_class.unsqueeze(1).to(device)
            
            # get text_embedding
            text_embeddings = torch.tensor([])
            for j in range(len(mask_class)):  # iterate this batch
                _class = int(mask_class[j].item())
                text_embedding = embedding_dict[_class]
                text_embeddings = torch.cat((text_embeddings, text_embedding), dim=0)
            text_embeddings = text_embeddings.to(device)\
            
            # forward
            predicted_dice = model(torch.cat((ct, pred_mask), dim=1), text_embeddings)
            
            # Compute loss
            loss1 = criterion1(predicted_dice, dice)
            loss2 = criterion2(predicted_dice, dice, text_embeddings)
            loss = loss1 + loss2
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_end = time.time()
            if (i+1) % info_interval == 0:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] MSE Loss: {:.4f} Rank Loss: {:.4f} Time: {:.3f}s".format(epoch+1, epochs, i+1, len(train_loader), loss1.item(), loss2.item(), iter_end-iter_start))
            iter_start = time.time()

        scheduler.step()

        if (epoch+1) % eval_interval == 0:
            model.eval()
            gt_dices = []
            pred_dices = []
            valid_loss = 0.0
            with torch.no_grad():
                for ct, pred_mask, gt_mask, dice, mask_class in valid_loader:
                    ct, pred_mask, gt_mask, dice, mask_class = ct.to(device), pred_mask.to(device), gt_mask.to(device), dice.unsqueeze(1).to(device), mask_class.unsqueeze(1).to(device)
                    
                    text_embeddings = torch.tensor([])
                    for j in range(len(mask_class)):
                        _class = int(mask_class[j].item())
                        text_embedding = embedding_dict[_class]
                        text_embeddings = torch.cat((text_embeddings, text_embedding), dim=0)
                    text_embeddings = text_embeddings.to(device)
                    
                    model_output = model(torch.cat((ct, pred_mask), dim=1))
                    predicted_dice = model_output  # resnet 1 dim output forward

                    # forward
                    predicted_dice = model(torch.cat((ct, pred_mask), dim=1), text_embeddings)
                    
                    gt_dices = gt_dices + list(dice.squeeze().cpu())
                    pred_dices = pred_dices + list(predicted_dice.squeeze().cpu())
                    
                    loss1 = criterion1(predicted_dice, dice)
                    loss2 = criterion2(predicted_dice, dice, text_embeddings)
                    loss = loss1 + loss2
                    
                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)
            
            corr_matrix = np.corrcoef(gt_dices, pred_dices)
            corr = corr_matrix[0, 1]
            print("LCC: {:.3f}".format(corr))
            
            if corr > max_val_coef:
                reached = epoch + 1
                max_val_coef = corr
                best_model = copy.deepcopy(model)
    
                print('best model saved')
                torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
            
            print(f"Validation Loss: {valid_loss:.4f}\n")

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(epochs, round(time.time() - start)))
    print('The max validation corr coef is: {:.4f}, reached at epoch {}.\n'.format(max_val_coef, reached))
    

if __name__ == "__main__":
    main()

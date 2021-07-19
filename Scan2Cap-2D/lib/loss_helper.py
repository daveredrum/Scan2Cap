import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_cap_loss(data_dict, weights):
    # unpack
    pred_caps = data_dict["lang_cap"]  # (B, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len"][0]
    target_caps = data_dict["lang_ids"][:, 1:num_words]  # (B, num_words - 1)
    _, _, num_vocabs = pred_caps.shape

    # caption loss
    # criterion = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor(weights).cuda())
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    cap_loss = F.cross_entropy(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1), ignore_index=0)

    # caption acc
    pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # B * (num_words - 1)
    target_caps = target_caps.reshape(-1)  # B * (num_words - 1)
    masks = target_caps != 0
    masked_pred_caps = pred_caps[masks]
    masked_target_caps = target_caps[masks]
    cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()

    return cap_loss, cap_acc


def get_scene_cap_loss(data_dict, weights):
    cap_loss, cap_acc = compute_cap_loss(data_dict, weights)

    # store
    data_dict["cap_loss"] = cap_loss
    data_dict["cap_acc"] = cap_acc

    # Final loss function
    loss = data_dict["cap_loss"]
    data_dict["loss"] = loss

    return data_dict

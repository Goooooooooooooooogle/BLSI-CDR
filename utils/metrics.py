import torch
import numpy as np

from tqdm import tqdm


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, cuda):
    HR_s, HR_t,NDCG_s, NDCG_t = [],[],[],[]
    i = 0
    for user,item_s,item_t,sl,tl in test_loader:
        i +=1
        if cuda:
            user = user.cuda()
            item_s = item_s.cuda()
            item_t = item_t.cuda()
        
        predictions_s, predictions_t = model(user, item_s,item_t)
        _, indices_s = torch.topk(predictions_s, top_k)
        recommends_s = torch.take(item_s, indices_s).tolist()
        _, indices_t = torch.topk(predictions_t, top_k)
        recommends_t = torch.take(item_t, indices_t).tolist()
        
        gt_item_s = item_s[-1].item()
        gt_item_t = item_t[-1].item()
        HR_s.append(hit(gt_item_s, recommends_s))
        NDCG_s.append(ndcg(gt_item_s, recommends_s))
        
        HR_t.append(hit(gt_item_t, recommends_t))
        NDCG_t.append(ndcg(gt_item_t, recommends_t))

    return np.mean(HR_s), np.mean(NDCG_s), np.mean(HR_t), np.mean(NDCG_t)

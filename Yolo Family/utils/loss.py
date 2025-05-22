 
import torch
import torch.nn.functional as F

def yolo_v1_loss(pred, target, S=7, B=2, C=20,
                 λ_coord=5.0, λ_noobj=0.5, img_size=448):
    """
    pred:   (BATCH, S, S, C + B*5)  raw outputs
    target: list of dicts with keys 'boxes' (N×4 in absolute coords) and 'labels' (N,)
    """
    N = pred.shape[0]
    # --- 1) split predictions ---
    # class scores
    p_class = F.softmax(pred[..., :C], dim=-1)           # (N,S,S,C)
    # box predictions: raw x,y -> sigmoid, raw w,h -> exp, raw conf -> sigmoid
    p_box = pred[..., C:].view(N, S, S, B, 5)
    x = torch.sigmoid(p_box[..., 0])                     # (N,S,S,B)
    y = torch.sigmoid(p_box[..., 1])
    w = torch.exp(p_box[..., 2])
    h = torch.exp(p_box[..., 3])
    conf = torch.sigmoid(p_box[..., 4])

    # create grid offset for x,y
    # grid_x[i,j] = j/S, grid_y[i,j] = i/S
    cell_idx = torch.arange(S, device=pred.device).float() / S
    grid_y, grid_x = torch.meshgrid(cell_idx, cell_idx)
    grid_x = grid_x.unsqueeze(0).unsqueeze(-1)  # (1,S,S,1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1)

    # convert x,y from sigmoid offset to absolute [0,1]
    x_abs = (x + grid_x)     # still in [0,1]
    y_abs = (y + grid_y)

    # containers for losses
    loss_coord = 0.0
    loss_conf_obj = 0.0
    loss_conf_noobj = 0.0
    loss_class = 0.0

    for b in range(N):
        gt_boxes = target[b]['boxes']   # (M,4) absolute pixel coords
        gt_labels = target[b]['labels'] # (M,)
        M = gt_boxes.shape[0]
        if M == 0:
            # all S×S×B confidences get no‑obj penalty
            loss_conf_noobj += (conf[b]**2).sum()
            continue

        # normalize GT to [0,1]
        gt_norm = gt_boxes / img_size

        # compute which cell each GT center falls in
        cx, cy = gt_norm[:,0], gt_norm[:,1]
        cell_i = (cy * S).long().clamp(0, S-1)  # row
        cell_j = (cx * S).long().clamp(0, S-1)  # col

        # local offsets within cell
        local_x = (cx * S) - cell_j.float()
        local_y = (cy * S) - cell_i.float()

        # for each GT, pick the best of the B predictors (highest IoU)
        # build preds for IoU in pixels
        pred_boxes_pixel = []
        for k in range(B):
            # convert back to pixels
            bx = x_abs[b, cell_i, cell_j, k] * img_size
            by = y_abs[b, cell_i, cell_j, k] * img_size
            bw = w[b, cell_i, cell_j, k] * img_size
            bh = h[b, cell_i, cell_j, k] * img_size
            # (x1,y1,x2,y2)
            corners = torch.stack([
                bx - bw/2, by - bh/2,
                bx + bw/2, by + bh/2
            ], dim=-1)
            pred_boxes_pixel.append(corners)
        pred_boxes_pixel = torch.stack(pred_boxes_pixel, dim=1)  # (M, B, 4)

        # compute IoUs
        # (M, B)
        ious = compute_iou(pred_boxes_pixel.view(-1,4),
                           gt_boxes.repeat(B,1)).view(M,B)
        best_k = torch.argmax(ious, dim=1)  # (M,)

        # now compute losses for each GT:
        for m in range(M):
            i, j, k = cell_i[m], cell_j[m], best_k[m]
            # coordinate loss
            loss_coord += (local_x[m] - x[b,i,j,k])**2 + (local_y[m] - y[b,i,j,k])**2
            loss_coord += (torch.sqrt(gt_norm[m,2]) - torch.sqrt(w[b,i,j,k]))**2
            loss_coord += (torch.sqrt(gt_norm[m,3]) - torch.sqrt(h[b,i,j,k]))**2
            # object confidence
            loss_conf_obj += (conf[b,i,j,k] - 1)**2
            # class loss (only once per cell)
            loss_class += torch.sum((p_class[b,i,j] - F.one_hot(gt_labels[m], C).float())**2)
            # no‑object for the other predictor in same cell
            other_k = 1-k
            loss_conf_noobj += λ_noobj * (conf[b,i,j,other_k] - 0)**2

        # all cells with no GT => no‑obj for all B
        # mask out the cells that had GT
        mask = torch.ones((S,S,B), device=pred.device, dtype=torch.bool)
        mask[cell_i, cell_j, best_k] = False
        loss_conf_noobj += λ_noobj * (conf[b][mask]**2).sum()

    # scale & average
    loss = (λ_coord * loss_coord + loss_class + loss_conf_obj + loss_conf_noobj) / N
    return loss

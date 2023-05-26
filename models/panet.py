import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PANet(nn.Module):
    """
    Few-shot semantic segmentation model with prototype alignment, based on:

    Wang, K. et al. (2019), "PANet: Few-Shot Image Semantic Segmentation With
    Prototype Alignment". In: The IEEE International Conference on Computer Vision (ICCV).

    Source code:
    https://github.com/kaixin96/PANet
    """
    def __init__(self,
        n_way: int,
        k_shot: int,
        encoder: nn.Module,
        prototype_alignment: bool = True,
    ):
        """
        Args:
            n_way: Number of few-shot classes (labels).
            k_shot: Number of support samples per class.
            encoder: Initialised feature encoder.
            prototype_alignment: If True, computes the prototype alignment loss
                in addition to the query loss.
        """
        super().__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.encoder = encoder
        self.prototype_alignment = prototype_alignment
        self.device = torch.device('cuda')

        self.scaler = 20.0 # Multiplying factor for cosine distance from original paper


    def forward(self,
        supp_img: torch.Tensor, supp_mask: torch.Tensor, qry_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            supp_img:  Support images,
                torch.Tensor of shape (batch_size, n_way, k_shot, ch, H, W)
            supp_mask:  Binary segmentation masks for support images,
                torch.Tensor of shape (batch_size, n_way, k_shot, 1, H, W)
            qry_img:  Query images,
                torch.Tensor of shape (batch_size, N, ch, H, W),
                N = number of query images in each batch

        Return:
            output:  Model prediction,
                torch.Tensor of shape (batch_size x N, 1+n_way, H, W)
            align_loss:  Mean prototype alignment loss
            threshold_loss:  A zero-value tensor, not used in this model
        """

        batch_size = supp_img.shape[0]
        img_shape = supp_img.shape[-2:]
        ch = supp_img.shape[-3]

        # Get foreground and background masks
        # (in binary segmentation, background is just inverted foreground)
        fg_mask = supp_mask.view(batch_size, -1, 1, *img_shape)
        bg_mask = (~ fg_mask.type(torch.bool)).type(fg_mask.dtype)

        # Extract features from encoder network
        img_concat = torch.cat(
            [supp_img.view(-1, ch, *img_shape), qry_img.view(-1, ch, *img_shape)]
        )
        encoder_out = self.encoder(img_concat)

        # Separate and reshape support and query features 
        fts_size = encoder_out.shape[-3:]    # shape (hid_dims, H', W')  
        supp_fts = encoder_out[:batch_size * self.n_way * self.k_shot].view(
            batch_size, -1, *fts_size
        )
        qry_fts = encoder_out[batch_size * self.n_way * self.k_shot:].view(
            batch_size, -1, *fts_size
        )

        # Get predicted segmention score for all query features
        output = []
        align_loss = torch.Tensor([0]).to(self.device)
        for b in range(batch_size):

            # Get foreground and background prototypes via masked average pooling
            fg_prototype = self.get_prototypes(supp_fts[b], fg_mask[b], is_bg=False)
            bg_prototype = self.get_prototypes(supp_fts[b], bg_mask[b], is_bg=True)
            # Prototypes shape (1 + n_way, hid_dims)
            prototypes = torch.cat([bg_prototype, fg_prototype])  

            # Compute cosine distance between query features and prototypes
            # Shape (N, 1 + n_way, H', W')
            pred = self.get_distance(qry_fts[b], prototypes)

            output.append(F.interpolate(pred, size=img_shape, mode='bilinear'))


            # Calculate prototype alignment loss
            if self.prototype_alignment:
                loss = self.alignment_loss(qry_fts[b], pred, supp_fts[b], fg_mask[b])
                align_loss += loss


        output = torch.stack(output).view(-1, 1 + self.n_way, *img_shape)
        t_loss = torch.Tensor([0]).to(self.device)  # dummy loss, not used in PANet

        return output, align_loss / batch_size, t_loss
        

    def get_prototypes(self, 
        fts: torch.Tensor, mask: torch.Tensor, is_bg: bool=False,
    ) -> torch.Tensor:
        """
        Computes class prototypes from support features via masked average pooling.

        Args:
            fts:  Support features, shape (n_way x k_shot, hid_dims, H', W')
            mask:  Support foreground/background mask, shape (n_way x k_shot, 1, H, W)
            is_bg:  If True, average prototypes over classes to get a single
                background prototype
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_avg_pool = torch.divide(
            torch.sum(fts * mask, dim=(-2, -1)),
            torch.sum(mask, dim=(-2, -1)) + 1e-5
        )   # shape (n_way x k_shot, hid_dims)

        masked_avg_pool = masked_avg_pool.view(self.n_way, self.k_shot, -1)

        if is_bg:
            # Average over support features and classes, shape (1, hid_dims)
            prototype = torch.mean(masked_avg_pool, dim=(0,1)).unsqueeze(0)
        else:
            # Average over support features for each class, shape (n_way, hid_dims)
            prototype = torch.mean(masked_avg_pool, dim=1) 

        return prototype


    def get_distance(self, fts: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Returns a similarity score based on the cosine distance between features
        and prototypes.

        Args:
            fts:  Either query features, shape (N, hid_dims, H', W')
                  or support features, shape (k_shot, hid_dims, H', W')
            prototypes:  Class prototypes, shape (len(prototypes), hid_dims)
        """

        dist = F.cosine_similarity(
                    fts.unsqueeze(1),
                    prototypes.view(1, len(prototypes), -1, 1, 1),
                    dim=2
        ) * self.scaler

        return dist


    def alignment_loss(self,
        qry_fts: torch.Tensor, pred: torch.Tensor, supp_fts: torch.Tensor, fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean prototype alignment loss between support and query prediction.

        Args:
            qry_fts:  Query features, shape (N, hid_dims, H', W')
            pred:  Predicted segmentation scores, shape (N, 1 + n_way, H', W')
            supp_fts:  Support features, shape (n_way x k_shot, hid_dims, H', W')
            fg_mask:  Support foreground masks, shape (n_way x k_shot, 1, H, W)
        """

        # Estimate binary masks for each class from predicted segmentation score
        pred_mask = pred.argmax(dim=1, keepdim=True)  # argmax=0 for background
        binary_masks = [pred_mask == i for i in range(1 + self.n_way)] 

        # Consider only non-zero masks (plus background)
        nonzero_idxs = [0,] + [    
            i for i in range(1, 1 + self.n_way) if binary_masks[i].sum() != 0
        ]

        pred_mask = torch.stack(                 # shape (N, M, 1, H', W')
            binary_masks, dim=1                  # M = number of non-zero masks
        ).type(qry_fts.dtype)[:, nonzero_idxs]

        # Get query prototypes from masked average pooling, shape (M, hid_dims)
        qry_prototypes = torch.divide(
            torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(-2, -1)),
            torch.sum(pred_mask, dim=(-2, -1)) + 1e-5
        ).mean(0)


        # Compute the alignment loss (class-wise)
        loss = torch.Tensor([0]).to(self.device)
        supp_fts = supp_fts.view(self.n_way, -1, *supp_fts.shape[-3:])
        fg_mask = fg_mask.view(self.n_way, -1, *fg_mask.shape[-2:]).long()

        for idx, way in enumerate(nonzero_idxs[1:]):

            prototypes = qry_prototypes[[0, idx + 1]]  # [background, foreground]

            supp_pred = F.interpolate(
                self.get_distance(supp_fts[way-1], prototypes),
                size = fg_mask.shape[-2:],
                mode = 'bilinear'
            ) # shape (k_shot, 2, H, W)

            loss += F.cross_entropy(supp_pred, fg_mask[way-1]) / (len(nonzero_idxs)-1)

        return loss 
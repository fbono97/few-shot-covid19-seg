import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ALPNet(nn.Module):
    """
    Few-shot semantic segmentation model with prototype alignment and adaptive
    local prototype pooling, based on:

    Ouyang, C. et al. (2020), "Self-supervision with Superpixels: Training
    Few-Shot Medical Image Segmentation Without Annotation".
    In: European Conference on Computer Vision, Springer (2020), pp. 762-780.

    Source code:
    https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
    """
    def __init__(self,
        n_way: int,
        k_shot: int,
        encoder: nn.Module,
        pool_window: Tuple[int],
        fg_threshold: float,
        bg_threshold: float,
        prototype_alignment: bool = True,
    ):
        """
        Args:
            n_way: Number of few-shot classes (labels).
            k_shot: Number of support samples per class.
            encoder: Initialised feature encoder.
            pool_window: Kernel size of average pooling layer to generate local
                prototypes.
            fg_threshold: Foreground mask density threshold to accept local prototype.
            bg_threshold: Background mask density threshold to accept local prototype.
            prototype_alignment: If True, computes the prototype alignment loss
                in addition to the query loss.
        """
        super().__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.encoder = encoder
        self.fg_thresh = fg_threshold
        self.bg_thresh = bg_threshold
        self.prototype_alignment = prototype_alignment
        self.device = torch.device('cuda')
        self.scaler = 20.0 # Multiplying factor for cosine distance from original paper

        # Pooling operation to generate local prototypes
        self.pooling_layer = nn.Sequential(nn.AvgPool2d(pool_window),
                                           nn.Flatten(-2,-1))


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
            batch_size, self.n_way, self.k_shot, *fts_size
        )
        qry_fts = encoder_out[batch_size * self.n_way * self.k_shot:].view(
            batch_size, -1, *fts_size
        )

        # Get predicted segmention score for all query features
        output = []
        align_loss = torch.Tensor([0]).to(self.device)
        for b in range(batch_size):

            # Interpolate support masks to the same size as support features
            res_fg_mask = F.interpolate(
                fg_mask[b].float(), size=fts_size[1:], mode='bilinear'
            ).view(self.n_way, self.k_shot, 1, *fts_size[1:])

            res_bg_mask = F.interpolate(
                bg_mask[b].float(), size=fts_size[1:], mode='bilinear'
            ).view(self.n_way, self.k_shot, 1, *fts_size[1:])


            # Get foreground and background scores for each class
            fg_pred = []
            bg_pred = []
            for way in range(self.n_way):
                fg_pred.append(
                    self.get_class_pred(supp_fts[b, way], res_fg_mask[way],
                                        qry_fts[b], self.fg_thresh, is_bg=False)
                )
                bg_pred.append(
                    self.get_class_pred(supp_fts[b, way], res_bg_mask[way],
                                        qry_fts[b], self.bg_thresh, is_bg=True)
                )

            fg_pred = torch.stack(fg_pred, dim=1) 
            bg_pred = torch.stack(bg_pred, dim=1).mean(dim=1, keepdim=True)

            pred = torch.cat([bg_pred, fg_pred], dim=1)  # Shape (N, 1+n_way, H', W')
            output.append(F.interpolate(pred, size=img_shape, mode='bilinear'))


            # Calculate prototype alignment loss
            if self.prototype_alignment:
                loss = self.alignment_loss(qry_fts[b], pred, supp_fts[b], fg_mask[b])
                align_loss += loss


        output = torch.stack(output).view(-1, 1 + self.n_way, *img_shape)
        t_loss = torch.Tensor([0]).to(self.device)  # dummy loss, not used in ALPNet

        return output, align_loss / batch_size, t_loss


    def get_class_pred(self,
        supp_fts: torch.Tensor,
        supp_mask: torch.Tensor,
        qry_fts: torch.Tensor,
        thresh: float,
        is_bg: bool=False,
    ) -> torch.Tensor:
        """
        Generates local and global prototypes from support features and computes
        a similarity score between these prototypes and query features.

        Args:
            supp_fts:  Support features, shape (k_shot, hid_dims, H', W')
            supp_mask:  Support foreground/background mask, shape (k_shot, 1, H', W')
            qry_fts:  Query features, shape (N, hid_dims, H', W')
            thresh: Foreground/Background threshold
            is_bg:  If False, combine local and global prototypes, else use
                local prototypes only
        
        Note: the shapes of support and query samples are inverted when
        computing the prototype alignment loss.
        """

        # Compute a global class prototype, shape (1, hid_dims)
        glob_prototype = torch.divide(
            torch.sum(supp_fts * supp_mask, dim=(-2, -1)),
            torch.sum(supp_mask, dim=(-2, -1)) + 1e-5
        ).mean(dim=0, keepdim=True)


        # Average pooling of support features and masks,
        # shapes (k_shot, H'' x W'', hid_dims) and (k_shot, H'' x W'')
        pooled_fts = self.pooling_layer(supp_fts).permute(0,2,1)
        pooled_mask = self.pooling_layer(supp_mask).squeeze(1)

        # Compute local prototypes, shape (n_prototypes, hid_dims)
        local_prototypes = pooled_fts[pooled_mask > thresh, :]


        # If no local prototypes are available, predict with global only
        if local_prototypes.shape[0] < 1:
            pred = F.cosine_similarity(    # Shape (N, H', W')
                qry_fts,
                glob_prototype.view(1, -1, 1, 1),
                dim=1
            ) * self.scaler      

            return pred

        if is_bg:
            protos = local_prototypes
        else:
            # Combine local and global prototypes, shape (1+n_prototypes, hid_dims)
            protos = torch.cat([glob_prototype, local_prototypes])

        # Calculate class-wise similarity score for query fts, shape (N, H', W')
        pred = self.similarity_score(protos, qry_fts)

        return pred

    
    def similarity_score(self, protos: torch.Tensor, qry_fts: torch.Tensor) -> torch.Tensor:
        """
        Returns a similarity score based on the cosine distance between features
        and prototypes.

        Args:
            protos:  Local and global prototypes of current class, shape (len(protos), hid_dims)
            qry_fts:  Query features, shape (N, hid_dims, H', W')
        """

        # Compute prototype-wise cosine distance, shape (N, len(protos), H', W')
        proto_norm = torch.divide(protos,
                                  torch.norm(protos, p=2, dim=1, keepdim=True) + 1e-5)
        qry_norm = torch.divide(qry_fts,
                                torch.norm(qry_fts, p=2, dim=1, keepdim=True) + 1e-5)

        cos_dist = F.conv2d(qry_norm, proto_norm[..., None, None]) * self.scaler

        # Combine distances to get a similarity score for current class
        pred = torch.sum(F.softmax(cos_dist, dim = 1) * cos_dist, dim=1)

        return pred   # Shape (N, H', W')


    def alignment_loss(self,
        qry_fts: torch.Tensor, pred: torch.Tensor, supp_fts: torch.Tensor, fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean prototype alignment loss between support and query prediction.

        Args:
            qry_fts:  Query features, shape (N, hid_dims, H', W')
            pred:  Predicted segmentation scores, shape (N, 1 + n_way, H', W')
            supp_fts:  Support features, shape (n_way, k_shot, hid_dims, H', W')
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


        # Compute the alignment loss
        loss = torch.Tensor([0]).to(self.device)
        fg_mask = fg_mask.view(self.n_way, -1, *fg_mask.shape[-2:]).long()

        for idx, way in enumerate(nonzero_idxs[1:]):

            fg_pred = self.get_class_pred(qry_fts, pred_mask[:, idx+1],
                                          supp_fts[way-1], self.fg_thresh, is_bg=False)
            bg_pred = self.get_class_pred(qry_fts, pred_mask[:, 0],
                                          supp_fts[way-1], self.bg_thresh, is_bg=True)

            supp_pred = F.interpolate(torch.stack([bg_pred, fg_pred], dim=1),
                                      size=fg_mask.shape[-2:],
                                      mode='bilinear')   # shape (k_shot, 2, H, W)

            loss += F.cross_entropy(supp_pred, fg_mask[way-1]) / (len(nonzero_idxs)-1)

        return loss 

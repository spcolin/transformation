import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from core.utils import utils
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)     # B*2*H*W

        B, _, H, W = coords1.shape

        tmp_coord = utils.coords_grid(B, H, W).to(coords1.device)
        ones = torch.ones(B, 1, H, W).to(coords1.device)
        homo_coord = torch.cat([tmp_coord, ones], 1).permute(0, 2, 3, 1).unsqueeze(4).clone().detach()

        if flow_init is not None:
            coords1 = coords1 + flow_init

        predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_trans = self.update_block(net, inp, corr, flow)

            delta_trans = delta_trans.permute(0, 2, 3, 1).view(-1, H, W, 3, 3) * 0.003

            identity_trans = torch.eye(3).expand(B, H, W, 3, 3).to(coords1.device)

            final_trans = identity_trans + delta_trans

            transformed_coord = torch.matmul(final_trans, homo_coord.detach()).squeeze(4)  # B*46*62*3

            inhomo_coord = transformed_coord[:, :, :, :2] / transformed_coord[:, :, :, -1].unsqueeze(3)  # B*46*62*2


            delta_flow=inhomo_coord.permute(0,3,1,2)-coords0
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # coords1=inhomo_coord.permute(0,3,1,2)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                if self.args.loss_type=='flow':
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                    predictions.append(flow_up)
                else:
                    # coords_up=self.upsample_flow(coords1,up_mask)
                    # predictions.append(coords_up)

                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                    B_up, _, H_up, W_up=flow_up.shape
                    coords_up=flow_up+utils.coords_grid(B_up,H_up,W_up).to(flow_up.device)
                    predictions.append(coords_up)


        if test_mode:
            if self.args.loss_type=='flow':
                return coords1 - coords0, flow_up
            else:
                # B_up,_,H_up,W_up=coords_up.shape
                # return coords1-coords0,coords_up-utils.coords_grid(B_up,H_up,W_up).to(coords_up.device)
                return coords1 - coords0, flow_up

        return predictions

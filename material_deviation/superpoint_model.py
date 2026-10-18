# superpoint_module.py
"""
SuperPoint 特征提取模块
基于 demo_superpoint.py 中的实现
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Jet colormap for visualization.
myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])


class SuperPointNet(torch.nn.Module):
    """Pytorch definition of SuperPoint Network."""

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass that jointly computes unprocessed point and descriptor tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend:
    """Wrapper around pytorch net to help with pre and post image processing."""

    def __init__(self, weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()

        if cuda and torch.cuda.is_available():
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
            self.device = torch.device('cuda')
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
            self.device = torch.device('cpu')

        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T
        """
        grid = np.zeros((H, W)).astype(int)
        inds = np.zeros((H, W)).astype(int)

        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)

        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)

        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i

        # Pad the border of the grid.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')

        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1

        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """Process a numpy image to extract points and descriptors."""
        assert img.ndim == 2 and img.dtype == np.float32
        H, W = img.shape

        # Prepare input tensor
        inp = torch.from_numpy(img.copy()).view(1, 1, H, W).to(self.device)

        # Forward pass
        with torch.no_grad():
            outs = self.net.forward(inp)
            semi, coarse_desc = outs[0], outs[1]

            # Convert to numpy
            if self.cuda and torch.cuda.is_available():
                semi = semi.cpu().numpy().squeeze()
                coarse_desc = coarse_desc.cpu()
            else:
                semi = semi.numpy().squeeze()

        # Process points
        dense = np.exp(semi)
        dense = dense / (np.sum(dense, axis=0) + 1e-5)
        nodust = dense[:-1, :, :]

        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        heatmap = nodust.transpose(1, 2, 0).reshape(Hc, Wc, self.cell, self.cell)
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])

        xs, ys = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None

        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]

        # Apply NMS
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)

        # Sort by confidence
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]

        # Remove points along border
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        pts = pts[:, ~(toremoveW | toremoveH)]

        # Process descriptor
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2).float().to(self.device)

            with torch.no_grad():
                desc = F.grid_sample(coarse_desc, samp_pts)

            if self.cuda and torch.cuda.is_available():
                desc = desc.cpu().numpy()
            else:
                desc = desc.numpy()

            desc = desc.reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        return pts, desc, heatmap

    def run_scaled(self, img, target_size=(1920, 1080)):
        """
        对高分辨率图像进行缩放推理，然后将特征点坐标映射回原始尺寸。
        """
        assert img.ndim == 2 and img.dtype == np.float32

        # 记录原始图像尺寸
        original_H, original_W = img.shape
        target_W, target_H = target_size

        # 计算缩放比例
        scale_x = original_W / target_W
        scale_y = original_H / target_H

        # 缩放图像到目标尺寸
        scaled_img = cv2.resize(img, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

        # 在缩放后的图像上进行推理
        pts, desc, heatmap = self.run(scaled_img)

        # 将特征点坐标映射回原始尺寸
        if pts.shape[1] > 0:
            pts[0, :] = pts[0, :] * scale_x  # x坐标缩放
            pts[1, :] = pts[1, :] * scale_y  # y坐标缩放

        # 将热力图也映射回原始尺寸
        original_heatmap = cv2.resize(heatmap, (original_W, original_H), interpolation=cv2.INTER_LINEAR)

        return pts, desc, original_heatmap
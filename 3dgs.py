import torch


class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class CameraExtrinsics:
    def __init__(self, R, t):
        self.R = R
        self.t = t


class VanillaGaussian(torch.nn.Module):
    def __init__(self, means, scales, quats, opacities):
        super().__init__()
        # Ignore color-related fields.
        self._means = means  # (N, 3)
        self._scales = scales # (N, 3)
        self._quats = quats # (N, 4)
        self._opacities = opacities # (N,)

    @property
    def get_scaling(self):
        # Make sure the scaling is positive.
        return torch.exp(self._scales)

    @property
    def get_opacities(self):
        # Make sure the opacities are between 0 and 1.
        return torch.sigmoid(self._opacities)

    @property
    def get_quats(self):
        # Make sure the quats are normalized.
        return self._quats / torch.norm(self._quats, dim=-1, keepdim=True)

    @property
    def get_means(self):
        return self._means

    def _quat_to_rot(self):
        """Convert quaternion to rotation matrix."""
        q = self.get_quats
        
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        return torch.stack([
            1-2*(y**2+z**2), 2*(x*y-w*z),     2*(x*z+w*y),
            2*(x*y+w*z),     1-2*(x**2+z**2), 2*(y*z-w*x),
            2*(x*z-w*y),     2*(y*z+w*x),     1-2*(x**2+y**2)
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))

    def world_gaussian_to_image(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        """
        Project 3D Gaussians to 2D image plane.
        """
        # Step 1: Contruct covariance matrix for 3D Gaussian in world coordinate.
        means = self.get_means

        S = torch.diag_embed(self.get_scaling) # (N, 3, 3)
        R = self._quat_to_rot() # (N, 3, 3)
        sigma_world = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)

        # Step 2: Transform covariance matrix to camera coordinate.
        means_camera = (extrinsics.R[None] @ means[..., None]).squeeze(-1) + extrinsics.t[None]
        sigma_camera = extrinsics.R @ sigma_world @ extrinsics.R.T
        
        # Step 3: Camera Cooridnate to Image Plane.
        # Step 3.1 means_camera to image plane.
        u = intrinsics.fx * means_camera[..., 0] / means_camera[..., 2] + intrinsics.cx
        v = intrinsics.fy * means_camera[..., 1] / means_camera[..., 2] + intrinsics.cy
        z = means_camera[..., 2]
        center_2d = torch.stack([u, v], dim=-1) # (N, 2)
        
        # Step 3.2 sigma_camera to image plane.
        zeros = torch.zeros_like(z)
        J = torch.stack([
            intrinsics.fx / z, zeros, -intrinsics.fx * means_camera[..., 0] / z**2,
            zeros, intrinsics.fy / z, -intrinsics.fy * means_camera[..., 1] / z**2
        ], dim=-1).reshape(means_camera.shape[:-1] + (2, 3))
        sigma_2d = J @ sigma_camera @ J.transpose(1, 2)

        return center_2d, sigma_2d


if __name__ == "__main__":
    means = torch.randn(100, 3)
    scales = torch.randn(100, 3)
    quats = torch.randn(100, 4)
    opacities = torch.randn(100)
    vanilla_gaussian = VanillaGaussian(means, scales, quats, opacities)
    print(vanilla_gaussian.get_scaling.shape)
    print(vanilla_gaussian.get_opacities.shape)
    intrinsics = CameraIntrinsics(100, 100, 100, 100)
    extrinsics = CameraExtrinsics(torch.eye(3), torch.zeros(3))
    centers, sigmas = vanilla_gaussian.world_gaussian_to_image(intrinsics, extrinsics)
    print(centers.shape)
    print(sigmas.shape)

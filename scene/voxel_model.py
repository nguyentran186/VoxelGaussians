import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch_scatter


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class VoxelModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
    def __init__(self, sh_degree, optimizer_type="default", voxel_size=0.008):
        # Initialize parameters
        self.active_sh_degree = 0  # Placeholder for the active spherical harmonics degree
        self.optimizer_type = optimizer_type  # Type of optimizer (e.g., sparse_adam)
        self.max_sh_degree = sh_degree  # Maximum spherical harmonics degree
        self.voxel_size = voxel_size  # Voxel size (important for grid discretization)
        
        # Initialize tensors for Gaussian primitive parameters
        self._xyz = torch.empty(0)  # Positions (x, y, z) of Gaussian primitives
        self._xyz_disp = torch.empty(0)
        self._features_dc = torch.empty(0)  # Features associated with the Gaussian (e.g., color)
        self._features_rest = torch.empty(0)  # Additional features for the Gaussian primitives
        self._scaling = torch.empty(0)  # Scale (spread or size) of the Gaussian primitives
        self._rotation = torch.empty(0)  # Rotation matrix (3x3) for the Gaussian primitives
        self._opacity = torch.empty(0)  # Opacity of the Gaussian primitives
        
        # Additional tensors for training
        self.max_radii2D = torch.empty(0)  # 2D radii (may refer to the projected radius on the 2D image plane)
        self.xyz_gradient_accum = torch.empty(0)  # Accumulated gradient for positions (for optimization)
        self.denom = torch.empty(0)  # Denominators for some calculations (e.g., normalization or weight adjustments)
        
        # Optimizer setup
        self.optimizer = None  # Placeholder for the optimizer object
        self.percent_dense = 0  # Percentage of dense Gaussian primitives in the grid
        self.spatial_lr_scale = 0  # Learning rate scaling factor for spatial parameters
        
        # Setup auxiliary functions like for loss calculation or optimization
        self.max_vals = 0
        self.min_vals = 0
        self.center = 0
        self.scale_factor = 1
        self.setup_functions()
        
    def capture(self):
        return (
            self.active_sh_degree,  # Current spherical harmonics degree
            self._xyz,               # Tensor of Gaussian primitive positions (x, y, z)
            self._xyz_disp,
            self._features_dc,       # Tensor of color features for the Gaussian primitives
            self._features_rest,     # Tensor of additional features for the Gaussian primitives
            self._scaling,           # Tensor of scale values for the Gaussian primitives
            self._rotation,          # Tensor of rotation matrices for the Gaussian primitives
            self._opacity,           # Tensor of opacity values for the Gaussian primitives
            self.max_radii2D,        # 2D radii tensor (possibly for projection onto 2D space)
            self.xyz_gradient_accum, # Tensor of accumulated gradients for optimization
            self.denom,              # Tensor of denominators for calculations like normalization
            self.optimizer.state_dict() if self.optimizer else None,  # Optimizer state dict (if available)
            self.spatial_lr_scale,   # Learning rate scaling factor for spatial parameters
            self.percent_dense       # Percentage of dense Gaussian primitives in the grid (if tracked)
        )
        
    def restore(self, model_args, training_args):
        # Unpack the model arguments to assign the class attributes
        (
            self.active_sh_degree,       # Set the current spherical harmonics degree
            self._xyz,                   # Restore positions of the Gaussian primitives (x, y, z)
            self._xyz_disp,
            self._features_dc,           # Restore the color features of the Gaussian primitives
            self._features_rest,         # Restore other features of the Gaussian primitives
            self._scaling,               # Restore the scaling factors of the Gaussian primitives
            self._rotation,              # Restore the rotation matrices of the Gaussian primitives
            self._opacity,               # Restore the opacity values of the Gaussian primitives
            self.max_radii2D,            # Restore the 2D radii (if used for projection)
            xyz_gradient_accum,          # Restore the accumulated gradients for positions
            denom,                       # Restore the denominator values for normalization
            opt_dict,                    # Restore the optimizer's state dict
            self.spatial_lr_scale        # Restore the spatial learning rate scale
        ) = model_args

        # Call the training setup function with the provided arguments
        self.training_setup(training_args)

        # Assign the restored values to the instance variables
        self.xyz_gradient_accum = xyz_gradient_accum  # Assign the restored gradients for positions
        self.denom = denom                            # Assign the restored denominators for calculations

        # Load the optimizer state dict to restore the optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)  # Load optimizer state if it exists

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz + 1.5 * torch.tanh(self._xyz_disp) * self.voxel_size
    
    @property
    def get_original_xyz(self):
        return self._xyz
    
    @property
    def get_xyz_disp(self):
        return self._xyz_disp
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        
    def create_from_pcd(self, pcd, cam_infos: int, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        
        # Step 1: Normalize the point cloud into the unit cube [0, 1]^3
        pcd_points = np.asarray(pcd.points)
        self.min_vals = pcd_points.min(axis=0)
        self.max_vals = pcd_points.max(axis=0)
        self.center = (self.min_vals + self.max_vals) / 2
        
        # Translate point cloud to center it at origin
        pcd_points -= self.center

        # Scale the point cloud to fit into a unit cube
        self.scale_factor = 1.0 / np.max(self.max_vals - self.min_vals)  # Scale based on the largest dimension
        
        # Scale the points
        pcd_points *= self.scale_factor
        
        # Step 2: Downsample the point cloud with a voxel size of d = 0.008
        voxel_size = self.voxel_size
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        fused_point_cloud = torch.tensor(np.asarray(downpcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(downpcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Step 4: Displacement within voxel and compute scale
        displacement = ((torch.rand_like(fused_point_cloud) - 0.5) * voxel_size)
        displaced_points = fused_point_cloud + displacement
        
        # Step 5: Compute distances and scales
        dist2 = torch.clamp_min(distCUDA2(displaced_points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        voxel_indices = torch.floor(fused_point_cloud / voxel_size).float()
        
        self._voxel_idx = nn.Parameter(voxel_indices.requires_grad_(False))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(False))
        self._xyz_disp = nn.Parameter(displacement.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._xyz_disp], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_disp"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = (self.get_xyz / self.scale_factor + torch.tensor(self.center, dtype=torch.float, device="cuda")).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_disp, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "xyz_disp": new_disp,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_create_new_primitives(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def densify_and_create_new_primitives(self, grads, grad_threshold, scene_extent, N=8):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.abs(self._xyz_disp)) >= self.voxel_size / 2)
        
        abs_disp = torch.abs(self._xyz_disp)
        exceeds_voxel_size = abs_disp >= (self.voxel_size / 2)
        new_xyz = self._xyz[selected_pts_mask]
        unit_displacement = torch.sign(self._xyz_disp[selected_pts_mask]) * exceeds_voxel_size[selected_pts_mask].float()
        new_xyz = new_xyz + unit_displacement * self.voxel_size
        new_disp = torch.zeros_like(new_xyz)
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        
        # Step 1: Identify unique `xyz` points and group indices
        unique_xyz, inverse_indices = torch.unique(new_xyz, dim=0, return_inverse=True)

        # Step 2: Aggregate (mean) features for each unique `xyz` point
        # Using `torch_scatter.scatter_mean` for aggregation
        aggregated_disp = torch_scatter.scatter_mean(new_disp, inverse_indices, dim=0)
        aggregated_features_dc = torch_scatter.scatter_mean(new_features_dc, inverse_indices, dim=0)
        aggregated_features_rest = torch_scatter.scatter_mean(new_features_rest, inverse_indices, dim=0)
        aggregated_opacities = torch_scatter.scatter_mean(new_opacities, inverse_indices, dim=0)
        aggregated_scaling = torch_scatter.scatter_mean(new_scaling, inverse_indices, dim=0)
        aggregated_rotation = torch_scatter.scatter_mean(new_rotation, inverse_indices, dim=0)
        aggregated_tmp_radii = torch_scatter.scatter_mean(new_tmp_radii, inverse_indices, dim=0)

        # Step 3: Delete competing voxels, keeping only unique averaged results
        final_xyz = unique_xyz
        final_disp = aggregated_disp
        final_features_dc = aggregated_features_dc
        final_features_rest = aggregated_features_rest
        final_opacities = aggregated_opacities
        final_scaling = aggregated_scaling
        final_rotation = aggregated_rotation
        final_tmp_radii = aggregated_tmp_radii

        self.densification_postfix(final_xyz, final_disp, final_features_dc, final_features_rest, final_opacities, final_scaling, final_rotation, final_tmp_radii)
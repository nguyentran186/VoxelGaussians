import open3d as o3d

# Load the PLY file
ply_file_path = "/home/nguyen/code/TD3D/gaussian-splatting/data/statue/sparse/0/points3D.ply"  # Replace with your .ply file path
point_cloud = o3d.io.read_point_cloud(ply_file_path)

# Print some information
print(point_cloud.points[0])
print(point_cloud)

downpcd = point_cloud.voxel_down_sample(voxel_size=1)
print(downpcd.points[0])
print(downpcd)

# # Visualize the point cloud
breakpoint()

import bempp.api
import numpy as np
import os
import gmsh

def get_spheres(centers, radiis):
    val = []

    for r_k, c_k in zip(centers, radiis):
        grid = bempp.api.shapes.sphere(r = r_k, origin = c_k, h=0.1)
        val.append(grid)
    return val


def complute_Rodriguez_rotation_matrix(axis, angle):
    """Compute the Rodriguez rotation matrix for a given axis and angle."""
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
    ])
    
    return rotation_matrix


def select_geometry(geometry, center_list, epsilon=1.0, h=0.1, directions=None, angles=None):

    grid_list = []
    
    if directions is None:
        directions = [[1,0,0].copy() for _ in center_list]
        angles = [0 for _ in center_list]

    if geometry == 'ellipsoid':
        for center in center_list:
            grid = bempp.api.shapes.ellipsoid(r1 = 0.5*epsilon, r2 = 0.5*epsilon, r3 = epsilon, origin = center, h=h)
            grid_list.append(grid)
    elif geometry == 'unit_sphere':
        for center in center_list:
            grid = bempp.api.shapes.sphere(origin = center, h=h)
            grid_list.append(grid)
    elif geometry == 'ellipsoid_cut':
        for (center, d, angle)  in zip(center_list, directions, angles):
            # d = np.random.normal(size=3)
            # d = d / np.linalg.norm(d)  # Normalize the direction vector
            # angle = np.random.uniform(-np.pi/2, np.pi/2)
            R = complute_Rodriguez_rotation_matrix(d, angle)
            new_axis = np.dot(R, np.array([1, 0, 0])) 
            grid = bempp.api.shapes.ellipsoid_cut(r = epsilon, origin = center, h=h, d=d, angle=angle, new_axis=new_axis)
            grid_list.append(grid)
    else:
        raise ValueError(f"Geometry '{geometry}' is not supported.")
    return grid_list


def get_ellipsoids(center_list, a, b, c):
    return [(x0, y0, z0, a, b, c) for x0, y0, z0 in center_list]


def get_grid_points(N_grid, xmin, xmax, ymin, ymax, plane='XY'):
    Nx = Ny = N_grid
    # creates a matrix of size (3 x N_grid), where the columns of the matrix are coordinates (x,y,z)
    # 'XY', 'XZ' and 'YZ' correspond to the selected projected surfaces 
    if plane == 'XY':
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))) 
    elif plane == 'XZ':
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(), np.zeros(plot_grid[0].size), plot_grid[1].ravel())) 
    elif plane == 'YZ':
        plot_grid = np.mgrid[ymin:ymax:Ny * 1j, xmin:xmax:Nx * 1j]
        points = np.vstack((np.zeros(plot_grid[0].size), plot_grid[1].ravel(), plot_grid[0].ravel())) 
    return points # return (N_grid^2, 3) array of points

def get_grid_indices(points, geometry, center, epsilon, plane='XY'):
    idx = np.ones_like(points[0], dtype=bool)
    if geometry == 'ellipsoid':
        x0, y0, z0, a, b, c = center[0], center[1], center[2], 0.5, 0.5, 1
        if plane == 'XY':
            x, y, z = points
            geometry_condition= ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 > epsilon ** 2
            idx &= geometry_condition

        elif plane == 'XZ':
            x, y, z = points
            geometry_condition= ((x - x0) / a) ** 2 + ((z - z0) / c) ** 2 > epsilon ** 2
            idx &= geometry_condition 

        elif plane == 'YZ':
            x, y, z = points
            geometry_condition= ((y - y0) / b) ** 2 + ((z - z0) / c) ** 2 > epsilon ** 2
            idx &= geometry_condition
    elif geometry == 'unit_sphere':
        x0, y0, z0 = center[0], center[1], center[2]

        if plane == 'XY':
            x, y, z = points
            geometry_condition= (x - x0) ** 2 + (y - y0) ** 2 > epsilon ** 2
            idx &= geometry_condition 

        elif plane == 'XZ':
            x, y, z = points
            geometry_condition= (x - x0) ** 2 + (z - z0) ** 2 > epsilon ** 2
            idx &= geometry_condition 

        elif plane == 'YZ':
            x, y, z = points
            geometry_condition= (y - y0) ** 2 + (z - z0) ** 2 > epsilon ** 2
            idx &= geometry_condition

    elif geometry == 'ellipsoid_cut':
        x0, y0, z0 = c[0], c[1], c[2]

        if plane == 'XY':
            x, y, z = points
            geometry_condition= (((x - x0) / 1) ** 2 + ((y - y0) / 0.5) ** 2 > 1.0) | (x < -0.5) | (((x - x0) / 0.9) ** 2 + ((y - y0) / 0.4) ** 2 < 1.0)
            idx &= geometry_condition 

        elif plane == 'XZ':
            x, y, z = points
            geometry_condition= geometry_condition = (((x - x0) / 1) ** 2 + ((z - z0) / 0.5) ** 2 > 1.0) | (x < -0.5) | (((x - x0) / 0.9) ** 2 + ((z - z0) / 0.4) ** 2 < 1.0)
            idx &= geometry_condition 

        elif plane == 'YZ':
            x, y, z = points
            geometry_condition = ((y - y0) ** 2 + (z - z0) ** 2 < 0.4 ** 2) | ((y - y0) ** 2 + (z - z0) ** 2 > 0.5 ** 2) 
            idx &= geometry_condition

    elif geometry == '':
        pass
    pass

    return idx


def get_grid(geometry, center_list, N_grid, xmin, xmax, ymin, ymax, epsilon = 1.0, plane='XY'):
    '''

    Creates a matrix of size (3 x N_grid^2), where the columns of the matrix are coordinates (x,y,z)
    
    'XY', 'XZ' and 'YZ' correspond to the selected projected surfaces 

    '''
    Nx = Ny = N_grid
    
    if plane == 'XY':
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))) 
    elif plane == 'XZ':
        plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
        points = np.vstack((plot_grid[0].ravel(), np.zeros(plot_grid[0].size), plot_grid[1].ravel())) 
    elif plane == 'YZ':
        plot_grid = np.mgrid[ymin:ymax:Ny * 1j, xmin:xmax:Nx * 1j]
        points = np.vstack((np.zeros(plot_grid[0].size), plot_grid[1].ravel(), plot_grid[0].ravel())) 

    idx = np.ones_like(points[0], dtype=bool) # array of 'True' [True, True, True,...,True]
    
    if geometry == 'ellipsoid':
        ellipsoids = get_ellipsoids(center_list, 0.5, 0.5, 1)

        for x0, y0, z0, a, b, c in ellipsoids:
            if plane == 'XY':
                x, y, z = points
                geometry_condition= ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 > epsilon ** 2
                idx &= geometry_condition# 

            elif plane == 'XZ':
                x, y, z = points
                geometry_condition= ((x - x0) / a) ** 2 + ((z - z0) / c) ** 2 > epsilon ** 2
                idx &= geometry_condition 

            elif plane == 'YZ':
                x, y, z = points
                geometry_condition= ((y - y0) / b) ** 2 + ((z - z0) / c) ** 2 > epsilon ** 2
                idx &= geometry_condition
    elif geometry == 'unit_sphere':
        
        for c in center_list:
            x0, y0, z0 = c[0], c[1], c[2]

            if plane == 'XY':
                x, y, z = points
                geometry_condition= (x - x0) ** 2 + (y - y0) ** 2 > epsilon ** 2
                idx &= geometry_condition 

            elif plane == 'XZ':
                x, y, z = points
                geometry_condition= (x - x0) ** 2 + (z - z0) ** 2 > epsilon ** 2
                idx &= geometry_condition 

            elif plane == 'YZ':
                x, y, z = points
                geometry_condition= (y - y0) ** 2 + (z - z0) ** 2 > epsilon ** 2
                idx &= geometry_condition

    elif geometry == 'ellipsoid_cut':
        
        for c in center_list:
            x0, y0, z0 = c[0], c[1], c[2]

            if plane == 'XY':
                x, y, z = points
                geometry_condition= (((x - x0) / 1) ** 2 + ((y - y0) / 0.5) ** 2 > 1.0) | (x < -0.5) | (((x - x0) / 0.9) ** 2 + ((y - y0) / 0.4) ** 2 < 1.0)
                idx &= geometry_condition 

            elif plane == 'XZ':
                x, y, z = points
                geometry_condition= geometry_condition = (((x - x0) / 1) ** 2 + ((z - z0) / 0.5) ** 2 > 1.0) | (x < -0.5) | (((x - x0) / 0.9) ** 2 + ((z - z0) / 0.4) ** 2 < 1.0)
                idx &= geometry_condition 

            elif plane == 'YZ':
                x, y, z = points
                geometry_condition = ((y - y0) ** 2 + (z - z0) ** 2 < 0.4 ** 2) | ((y - y0) ** 2 + (z - z0) ** 2 > 0.5 ** 2) 
                idx &= geometry_condition

    elif geometry == '':
        pass

    return points, idx 


def generate_points_on_sphere(n, radius=1.0):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.pi * (3. - np.sqrt(5))  # Golden angle

    y = 1 - (indices / n) * 2  # y goes from 1 to -1
    r = np.sqrt(1 - y**2)      # radius at each y

    theta = phi * indices
    x = np.cos(theta) * r
    z = np.sin(theta) * r

    points = np.stack((x, y, z), axis=1) * radius
    return points


def generate_points_on_circle(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x) 

    points = np.stack((x, y, z), axis=1)
    return points


def merge_and_save_meshes(obstacle_list):
    input_mesh_paths = []
    output_mesh_path = "merged_mesh.vtk"

    N = len(obstacle_list)
    for i in range(1,N+1):
        bempp.api.export("grid" + str(i) + ".msh", obstacle_list[i-1])
        input_mesh_paths.append("grid" + str(i) + ".msh")
    gmsh.initialize()

    try:
        gmsh.open(input_mesh_paths[0])
        for mesh_path in input_mesh_paths[1:]:
            gmsh.merge(mesh_path)
        gmsh.write(output_mesh_path)
    finally:
        gmsh.finalize()
    for mesh_path in input_mesh_paths:
        os.remove(mesh_path)

    grid = bempp.api.import_grid(output_mesh_path)
    return grid 

def merge_grids(obstacle_list):
    input_mesh_paths = []
    output_mesh_path = "merged_mesh.vtk"

    N = len(obstacle_list)
    for i in range(1,N+1):
        bempp.api.export("grid" + str(i) + ".msh", obstacle_list[i-1])
        input_mesh_paths.append("grid" + str(i) + ".msh")
    gmsh.initialize()

    try:
        gmsh.open(input_mesh_paths[0])
        for mesh_path in input_mesh_paths[1:]:
            gmsh.merge(mesh_path)
        gmsh.write(output_mesh_path)
    finally:
        gmsh.finalize()
    for mesh_path in input_mesh_paths:
        os.remove(mesh_path)

    grid = bempp.api.import_grid(output_mesh_path)
    return grid 
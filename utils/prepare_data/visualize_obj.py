import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_obj(obj_path):
    # Load the OBJ file
    mesh = trimesh.load(obj_path, process=False)

    # Print the number of vertices and faces
    print(f"Number of Vertices: {mesh.vertices.shape[0]}")
    print(f"Number of Faces: {mesh.faces.shape[0]}")

    # Plot the 3D model
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], c='b', marker='.')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    obj_path = "//home/chris/Code/PointClouds/data/other_files/00000050_80d90bfdd2e74e709956122a_trimesh_000.obj"  # Replace with the path to your OBJ file
    visualize_obj(obj_path)
import numpy as np
import meshio

class PointCloud3d(object):
    """Class representing a voxelized point cloud"""

    def __init__(self, position=None, color=None, filename=None):
        self.count = 0
        self.position = position  # Nx3 float32 np array
        self.color = color  # Nx3 uint8 np array
        self.morton_code = None  # length-N uint64 np array
        if position:
            self.count = len(position)
        if filename:
            self.read(filename)

    def read(self, filename):
        """Read voxelized point cloud from file"""

        mesh = meshio.read(filename)
        self.position = np.asarray(mesh.points, dtype=np.float32)
        self.count = len(self.position)
        red = np.asarray(mesh.point_data['red'], dtype=np.uint8)
        green = np.asarray(mesh.point_data['green'], dtype=np.uint8)
        blue = np.asarray(mesh.point_data['blue'], dtype=np.uint8)
        self.color = np.stack([red, green, blue], axis=1)
        assert len(self.color) == self.count

    def write(self, filename):
        """Write voxelized point cloud to file"""

        position = np.asarray(self.position, dtype=np.float32)
        color = np.rint(self.color).astype(np.uint8)
        red = self.color[:, 0]
        green = self.color[:, 1]
        blue = self.color[:, 2]
        point_data = {'red': red, 'green': green, 'blue': blue}
        mesh = meshio.Mesh(points=position, cells=[], point_data=point_data)
        mesh.write(filename)

    def mortonize_and_sort(self):
        """Compute and sort positions and colors by Morton code"""

        self.morton_code = MortonFromPosition(self.position)
        sort_arg = np.argsort(self.morton_code)
        self.position = self.position[sort_arg, :]
        self.color = self.color[sort_arg, :]


def MortonFromPosition(position):
    """Convert integer (x,y,z) positions to Morton codes

    Args:
      positions: Nx3 np array (will be cast to int32)

    Returns:
      Length-N int64 np array
    """

    position = np.asarray(position, dtype=np.int32)
    morton_code = np.zeros(len(position), dtype=np.int64)
    coeff = np.asarray([4, 2, 1], dtype=np.int64)
    for b in range(21):
        morton_code |= ((position & (1 << b)) << (2 * b)) @ coeff
    assert morton_code.dtype == np.int64
    return morton_code


def PositionFromMorton(morton_code):
    """Convert int64 Morton code to int32 (x,y,z) positions

    Args:
      morton_code: int64 np array

    Returns:
      Nx3 int32 np array
    """

    morton_code = np.asarray(morton_code, dtype=np.int64)
    position = np.zeros([len(morton_code), 3], dtype=np.int32)
    shift = np.array([2, 1, 0], dtype=np.int64)
    for b in range(21):
        position |= ((morton_code[:, np.newaxis] >> shift[np.newaxis, :]) >> (2 * b)
                     ).astype(np.int32) & (1 << b)
    assert position.dtype == np.int32
    return position

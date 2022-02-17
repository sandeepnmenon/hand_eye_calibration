from numbers import Number
import math
import numpy as np
import numpy.testing as npt

from .quaternion import Quaternion


def quat2euler(q):
    R = quat2mat(q)
    rz, ry, rx = mat2euler(R)

    return rx, ry, rz


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X
    wY = w*Y
    wZ = w*Z
    xX = x*X
    xY = x*Y
    xZ = x*Z
    yY = y*Y
    yZ = y*Z
    zZ = z*Z
    return np.array(
        [[1.0-(yY+zZ), xY-wZ, xZ+wY],
            [xY+wZ, 1.0-(xX+zZ), yZ-wX],
            [xZ-wY, yZ+wX, 1.0-(xX+yY)]])


def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = atan2(r12, r13)
            else:
                y = -np.pi/2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x


class DualQuaternion(object):
    """ Clifford dual quaternion denoted as dq = q_rot + epsilon * q_dual.

    Can be instantiated by:
    >>> dq = DualQuaternion(q_rot, q_dual)
    >>> dq = DualQuaternion.from_vector([qrx, qry, qrz, qrw, qtx, qty, qtz, qtw])
    >>> dq = DualQuaternion.from_pose(x, y, z, q_x, q_y, q_z, q_w)
    >>> dq = DualQuaternion.from_pose_vector([x, y, z, q_x, q_y, q_z, q_w])
    >>> dq = DualQuaternion(q_rot, q_dual)
    >>> # Given a [4x4] transformation matrix T.
    >>> dq = DualQuaternion.from_transformation_matrix(T)
    """
    dq = np.array([0., 0., 0., 1.0, 0., 0., 0., 0.]).T

    def __init__(self, q_rot, q_dual):
        for i in [q_rot, q_dual]:
            assert isinstance(
                i, Quaternion), "q_rot and q_dual should be quaternions."

        self.dq = np.array([0., 0., 0., 1.0, 0., 0., 0., 0.]).T

        # Assign real part (rotation).
        self.dq[0:4] = q_rot.q.copy()

        # Assign dual part (translation).
        self.dq[4:8] = q_dual.q.copy()

        self.assert_normalization()

    def __str__(self):
        return "[q_rot: {}, q_dual: {}]".format(np.str(self.q_rot), np.str(self.q_dual))

    def __repr__(self):
        return ("<Dual quaternion q_rot {} q_dual {}>").format(self.q_rot, self.q_dual)

    def __add__(self, other):
        """ Dual quaternion addition. """
        dq_added = self.dq + other.dq
        return DualQuaternion.from_vector(dq_added)

    def __sub__(self, other):
        """ Dual quaternion subtraction. """
        dq_sub = self.dq - other.dq
        return DualQuaternion.from_vector(dq_sub)

    def __mul__(self, other):
        """ Dual quaternion multiplication.

        The multiplication with a scalar returns the dual quaternion with all
        elements multiplied by the scalar.

        The multiplication of two dual quaternions dq1 and dq2 as:
        q1_rot * q2_rot + epsilon * (q1_rot * q2_trans + q1_trans * q2_rot),
        where dq1 and dq2 are defined as:
        dq1 = q1_rot + epsilon * q1_trans,
        dq2 = q2_rot + epsilon * q2_trans.
        """
        if isinstance(other, DualQuaternion):
            rotational_part = self.q_rot * other.q_rot
            translational_part = (self.q_rot * other.q_dual +
                                  self.q_dual * other.q_rot)
            return DualQuaternion(rotational_part.copy(), translational_part.copy())
        elif isinstance(other, Number):
            dq = self.dq.copy()
            dq_out = dq * np.float64(other)
            return DualQuaternion.from_vector(dq_out)
        else:
            assert False, ("Multiplication is only defined for scalars or dual " "quaternions.")

    def __rmul__(self, other):
        """ Scalar dual quaternion multiplication.

        The multiplication with a scalar returns the dual quaternion with all
        elements multiplied by the scalar.
        """
        if isinstance(other, Number):
            dq = self.dq.copy()
            dq_out = np.float64(other) * dq
            return DualQuaternion.from_vector(dq_out)
        else:
            assert False, ("Multiplication is only defined for scalars or dual " "quaternions.")

    def __truediv__(self, other):
        """ Quaternion division with either scalars or quaternions.

        The division with a scalar returns the dual quaternion with all
        translational elements divided by the scalar.

        The division with a dual quaternion returns dq = dq1/dq2 = dq1 * dq2^-1,
        hence other divides on the right.
        """
        # TODO(ff): Check if this is correct.
        print("WARNING: This might not be properly implemented.")
        if isinstance(other, DualQuaternion):
            return self * other.inverse()
        elif isinstance(other, Number):
            dq = self.dq.copy()
            dq_out = dq / np.float64(other)
            return DualQuaternion.from_vector(dq_out)
        else:
            assert False, "Division is only defined for scalars or dual quaternions."

    def __div__(self, other):
        """ Quaternion division with either scalars or quaternions.

        The division with a scalar returns the dual quaternion with all
        translational elements divided by the scalar.

        The division with a dual quaternion returns dq = dq1 / dq2 = dq1 * dq2^-1.
        """
        return self.__truediv__(other)

    def __eq__(self, other):
        """ Check equality. """
        if isinstance(other, DualQuaternion):
            return np.allclose(self.dq, other.dq)
        else:
            return False

    @classmethod
    def from_vector(cls, dq):
        dual_quaternion_vector = None
        if isinstance(dq, np.ndarray):
            dual_quaternion_vector = dq.copy()
        else:
            dual_quaternion_vector = np.array(dq)
        return cls(Quaternion(q=dual_quaternion_vector[0:4]),
                   Quaternion(q=dual_quaternion_vector[4:8]))

    @classmethod
    def from_pose(cls, x, y, z, rx, ry, rz, rw):
        """ Create a normalized dual quaternion from a pose. """
        qr = Quaternion(rx, ry, rz, rw)
        qr.normalize()
        qt = (Quaternion(x, y, z, 0) * qr) * 0.5
        return cls(qr, qt)

    @classmethod
    def from_pose_vector(cls, pose):
        """ Create a normalized dual quaternion from a pose vector. """
        return cls.from_pose(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix):
        q_rot = Quaternion.from_rotation_matrix(
            transformation_matrix[0:3, 0:3])

        pose_vec = np.zeros(7)
        pose_vec[3:7] = q_rot.q
        pose_vec[0:3] = transformation_matrix[0:3, 3]

        return cls.from_pose_vector(pose_vec)

    @classmethod
    def identity(cls):
        identity_q_rot = Quaternion(0., 0., 0., 1.)
        identity_q_dual = Quaternion(0., 0., 0., 0.)
        return cls(identity_q_rot, identity_q_dual)

    # def conjugate_translation(self):
    #   """ Dual quaternion translation conjugate. """
    #   return DualQuaternion(self.q_rot.conjugate(), -self.q_dual.conjugate())

    def conjugate(self):
        """ Dual quaternion multiplication conjugate. """
        return DualQuaternion(self.q_rot.conjugate(), self.q_dual.conjugate())

    def inverse(self):
        """ Dual quaternion inverse. """
        assert self.norm()[0] > 1e-8
        return DualQuaternion(self.q_rot.inverse(),
                              -self.q_rot.inverse() * self.q_dual * self.q_rot.inverse())

    def enforce_positive_q_rot_w(self):
        """ Enforce a positive real part of the rotation quaternion. """
        assert self.norm()[0] > 1e-8
        if self.q_rot.w < 0.0:
            self.dq = -self.dq

    def norm(self):
        """ The norm of a dual quaternion. """
        assert self.q_rot.norm() > 1e-8, (
            "Dual quaternion has rotational part equal to zero, hence the norm is"
            "not defined.")
        real_norm = self.q_rot.norm()
        dual_norm = np.dot(self.q_rot.q, self.q_dual.q) / real_norm
        return (real_norm, dual_norm)

    def is_normalized(self):
        real_part = np.absolute(self.norm()[0] - 1.0) < 1e-8
        dual_part = np.absolute(self.norm()[1]) < 1e-8
        return real_part and dual_part

    def assert_normalization(self):
        assert self.is_normalized, "Something went wrong, the dual quaternion is not normalized!"

    def normalize(self):
        """ Normalize the dual quaternion. """
        real_norm = self.q_rot.norm()
        self.dq[0:4] = self.q_rot.q / real_norm
        self.dq[4:8] = self.q_dual.q / real_norm
        self.assert_normalization()

    def scalar(self):
        """ The scalar part of the dual quaternion.

        Defined as: scalar(dq) := 0.5*(dq+dq.conjugate())
        """
        scalar_part = 0.5 * (self + self.conjugate())
        npt.assert_allclose(
            scalar_part.dq[[0, 1, 2, 4, 5, 6]], np.zeros(6), atol=1e-6)
        return scalar_part.copy()

    def screw_axis(self):
        """ The rotation, translation and screw axis from the dual quaternion. """
        rotation = 2. * np.degrees(np.arccos(self.q_rot.w))
        rotation = np.mod(rotation, 360.)

        if (rotation > 1.e-12):
            translation = -2. * self.q_dual.w / \
                np.sin(rotation / 2. * np.pi / 180.)
            screw_axis = self.q_rot.q[0:3] / \
                np.sin(rotation / 2. * np.pi / 180.)
        else:
            translation = 2. * \
                np.sqrt(np.sum(np.power(self.q_dual.q[0:3], 2.)))
            if (translation > 1.e-12):
                screw_axis = 2. * self.q_dual.q[0:3] / translation
            else:
                screw_axis = np.zeros(3)

        # TODO(ntonci): Add axis point for completeness

        return screw_axis, rotation, translation

    def passive_transform_point(self, point):
        """ Applies the passive transformation of the dual quaternion to a point.
        """
        # TODO(ff): Check if the rotation is in the right direction.
        point_dq = DualQuaternion.from_pose(
            point[0], point[1], point[2], 0, 0, 0, 1)
        dq_in_new_frame = self * point_dq
        return dq_in_new_frame.to_pose()[0:3]

    def active_transform_point(self, point):
        """ Applies the active transformation of the dual quaternion to a point.
        """
        return self.inverse().passive_transform_point(point)

    # TODO(ff): Implement translational velocity, rotational velocity.
    # See for instance:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3576712/pdf/fnbeh-07-00007.pdf

    def to_matrix(self):
        """ Returns a [4x4] transformation matrix. """
        self.normalize()
        matrix_out = np.identity(4)
        matrix_out[0:3, 0:3] = self.q_rot.to_rotation_matrix()
        matrix_out[0:3, 3] = self.to_pose()[0:3]
        return matrix_out.copy()

    def to_pose(self):
        """ Returns a [7x1] pose vector.

        In the form: pose = [x, y, z, qx, qy, qz, qw].T.
        """
        self.normalize()

        pose = np.zeros(7)
        q_rot = self.q_rot
        if (q_rot.w < 0.):
            q_rot = -q_rot
        translation = (2.0 * self.q_dual) * self.q_rot.conjugate()

        pose[0:3] = translation.q[0:3].copy()
        pose[3:7] = q_rot.q.copy()
        return pose.copy()

    def to_pose_euler(self):
        """ Returns a [6x1] pose vector.

        In the form: pose = [x, y, z, roll, pitch, yaw].T.
        """
        x, y, z, qx, qy, qz, qw = self.to_pose()
        roll, pitch, yaw = quat2euler([qx, qy, qz, qw])
        pose = np.zeros(6)
        pose[0:3] = [x, y, z]
        pose[3:6] = [math.degrees(roll), math.degrees(
            pitch), math.degrees(yaw)]

        return pose

    def copy(self):
        """ Copy dual quaternion. """
        return DualQuaternion.from_vector(self.dq)

    @property
    def q_rot(self):
        return Quaternion(q=self.dq[0:4])

    @property
    def q_dual(self):
        return Quaternion(q=self.dq[4:8])

    @property
    def r_x(self):
        return self.dq[0]

    @property
    def r_y(self):
        return self.dq[1]

    @property
    def r_z(self):
        return self.dq[2]

    @property
    def r_w(self):
        return self.dq[3]

    @property
    def t_x(self):
        return self.dq[4]

    @property
    def t_y(self):
        return self.dq[5]

    @property
    def t_z(self):
        return self.dq[6]

    @property
    def t_w(self):
        return self.dq[7]

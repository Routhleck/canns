canns.analyzer.data.cell_classification.utils.geometry
======================================================

.. py:module:: canns.analyzer.data.cell_classification.utils.geometry

.. autoapi-nested-parse::

   Geometry Utilities

   Functions for geometric calculations including ellipse fitting,
   distance computations, and polygon operations.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.utils.geometry.t


Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.utils.geometry.cart2pol
   canns.analyzer.data.cell_classification.utils.geometry.fit_ellipse
   canns.analyzer.data.cell_classification.utils.geometry.pol2cart
   canns.analyzer.data.cell_classification.utils.geometry.polyarea
   canns.analyzer.data.cell_classification.utils.geometry.squared_distance
   canns.analyzer.data.cell_classification.utils.geometry.wrap_to_pi


Module Contents
---------------

.. py:function:: cart2pol(x, y)

   Transform Cartesian coordinates to polar coordinates.

   :param x: X coordinates
   :type x: np.ndarray
   :param y: Y coordinates
   :type y: np.ndarray

   :returns: * **theta** (*np.ndarray*) -- Angle in radians, range [-π, π]
             * **rho** (*np.ndarray*) -- Radius (distance from origin)

   .. rubric:: Examples

   >>> x = np.array([1, 0, -1])
   >>> y = np.array([0, 1, 0])
   >>> theta, rho = cart2pol(x, y)
   >>> print(theta)  # [0, π/2, π]
   >>> print(rho)    # [1, 1, 1]

   .. rubric:: Notes

   Equivalent to MATLAB's cart2pol function.


.. py:function:: fit_ellipse(x, y)

   Least-squares fit of ellipse to 2D points.

   Implements the Direct Least Squares Fitting algorithm by Fitzgibbon et al. (1999).
   This is a robust method that includes scaling to reduce roundoff error and
   returns geometric parameters rather than quadratic form coefficients.

   :param x: X coordinates of points (1D array)
   :type x: np.ndarray
   :param y: Y coordinates of points (1D array)
   :type y: np.ndarray

   :returns: **params** -- Array of shape (5,) containing:
             [center_x, center_y, radius_x, radius_y, theta_radians]
             where theta is the orientation angle of the major axis
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # Generate points on an ellipse
   >>> t = np.linspace(0, 2*np.pi, 100)
   >>> cx, cy = 5, 3  # center
   >>> rx, ry = 4, 2  # radii
   >>> angle = np.pi/4  # rotation
   >>> x = cx + rx * np.cos(t) * np.cos(angle) - ry * np.sin(t) * np.sin(angle)
   >>> y = cy + rx * np.cos(t) * np.sin(angle) + ry * np.sin(t) * np.cos(angle)
   >>> params = fit_ellipse(x, y)
   >>> print(f"Fitted center: ({params[0]:.2f}, {params[1]:.2f})")
   >>> print(f"Fitted radii: ({params[2]:.2f}, {params[3]:.2f})")

   .. rubric:: Notes

   Based on fitEllipse.m from the MATLAB codebase.

   .. rubric:: References

   Fitzgibbon, A.W., Pilu, M., and Fisher, R.B. (1999).
   "Direct least-squares fitting of ellipses". IEEE T-PAMI, 21(5):476-480.
   http://research.microsoft.com/en-us/um/people/awf/ellipse/


.. py:function:: pol2cart(theta, rho)

   Transform polar coordinates to Cartesian coordinates.

   :param theta: Angle in radians
   :type theta: np.ndarray
   :param rho: Radius (distance from origin)
   :type rho: np.ndarray

   :returns: * **x** (*np.ndarray*) -- X coordinates
             * **y** (*np.ndarray*) -- Y coordinates

   .. rubric:: Examples

   >>> theta = np.array([0, np.pi/2, np.pi])
   >>> rho = np.array([1, 1, 1])
   >>> x, y = pol2cart(theta, rho)
   >>> print(x)  # [1, 0, -1]
   >>> print(y)  # [0, 1, 0]

   .. rubric:: Notes

   Equivalent to MATLAB's pol2cart function.


.. py:function:: polyarea(x, y)

   Compute area of a polygon using the shoelace formula.

   :param x: X coordinates of polygon vertices
   :type x: np.ndarray
   :param y: Y coordinates of polygon vertices
   :type y: np.ndarray

   :returns: **area** -- Area of the polygon (always positive)
   :rtype: float

   .. rubric:: Examples

   >>> # Unit square
   >>> x = np.array([0, 1, 1, 0])
   >>> y = np.array([0, 0, 1, 1])
   >>> area = polyarea(x, y)
   >>> print(f"Area: {area}")  # Should be 1.0

   >>> # Triangle
   >>> x = np.array([0, 1, 0.5])
   >>> y = np.array([0, 0, 1])
   >>> area = polyarea(x, y)
   >>> print(f"Area: {area}")  # Should be 0.5

   .. rubric:: Notes

   Based on MATLAB's polyarea function using the shoelace formula.
   The polygon can be specified in either clockwise or counter-clockwise order.


.. py:function:: squared_distance(X, Y)

   Compute squared Euclidean distance matrix between two sets of points.

   Efficiently computes all pairwise squared distances between points in X and Y
   using the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y

   :param X: First set of points, shape (d, n) where d is dimension, n is number of points
   :type X: np.ndarray
   :param Y: Second set of points, shape (d, m) where d is dimension, m is number of points
   :type Y: np.ndarray

   :returns: **D** -- Squared distance matrix, shape (n, m)
             D[i, j] = ||X[:, i] - Y[:, j]||^2
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # 2D points
   >>> X = np.array([[0, 1, 2], [0, 0, 0]])  # 3 points along x-axis
   >>> Y = np.array([[0, 0], [1, 2]])  # 2 points along y-axis
   >>> D = squared_distance(X, Y)
   >>> print(D)  # Distances from X points to Y points

   .. rubric:: Notes

   Based on sqDistance inline function from gridnessScore.m and findCentreRadius.m.
   Uses bsxfun-style broadcasting for efficiency.


.. py:function:: wrap_to_pi(angles)

   Wrap angles to the range [-π, π].

   :param angles: Angles in radians
   :type angles: np.ndarray

   :returns: **wrapped** -- Angles wrapped to [-π, π]
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> angles = np.array([0, np.pi, -np.pi, 3*np.pi, -3*np.pi])
   >>> wrapped = wrap_to_pi(angles)
   >>> print(wrapped)  # All in [-π, π]

   .. rubric:: Notes

   Equivalent to MATLAB's wrapToPi function.


.. py:data:: t


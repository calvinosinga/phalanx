"""
Many different functions need access to these names, adding them here prevents circular imports
"""
# containers constants
RADIUS = 'radius'
MASS = 'mass'
VR = 'vr'
VT = 'vt'
VEL = 'velocity'  # optionally update to ['vx', 'vy', 'vz']
PID = 'pid'
UPID = 'upid'
STATUS = 'status'

# scene constants
BACKGROUND_COLOR = 'background'
CAM_POS = 'camera_position'
CAM_FOC = 'camera_focus_point'
CAM_UP = 'camera_view_up'
CAM_ZOOM = 'camera_parallel_scale'
CAM_PROJ = 'camera_parallel_projection'
CAM_ANGLE = 'camera_angle'
INTERP_TYPE = 'linear'
INTERP_STEP = 'stepping'
VIEW_SIZE = 'view_size'
SCENE_KEY = '_scene'

# graphics style constants - all should be implemented in graphic._setStyle
COLOR = 'color'
OPACITY = 'opacity'
LSTYLE = 'linestyle'
LWIDTH = 'linewidth'
SIZE = 'size'
SHAPE = 'shape'
LABEL = 'label'
TIPSIZE = 'tipsize'
SHOW_POINTS = 'show_points'
LIGHTING = 'lighting'
# other graphics constants
START = 'start'
STOP = 'stop'


ALLOWED_LINESTYLES = {'solid', 'dashed', 'dotted'}
ALLOWED_MARKER_SHAPES = {'sphere', 'cube', 'cone', 'arrow', 'cylinder', 'point'}
ALLOWED_INTERP_TYPES = {'Linear', 'Spline', 'Boolean'}


INT_GRAPHIC_NAMES = {SCENE_KEY}
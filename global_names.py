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
FRAME_RATE = 'frame_rate'

SCENE_KEY = '_scene'

# graphics constants
COLOR = 'color'
OPACITY = 'opacity'
LSTYLE = 'linestyle'
LWIDTH = 'linewidth'
SIZE = 'size'
SHAPE = 'shape'
LABEL = 'label'
TIPSIZE = 'tipsize'
SHOW_POINTS = 'show_points'

ALLOWED_LINESTYLES = {'solid', 'dashed', 'dotted'}
ALLOWED_MARKER_SHAPES = {'sphere', 'cube', 'cone', 'arrow', 'cylinder', 'point'}



INT_GRAPHIC_NAMES = {SCENE_KEY}
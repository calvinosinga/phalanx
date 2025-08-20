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
COLOR_CYCLE = [
    (0.8509803921568627, 0.3254901960784314, 0.09803921568627451),  # red
    (0.09803921568627451, 0.3254901960784314, 0.8509803921568627),  # blue
    (0.4666666666666667, 0.6745098039215687, 0.18823529411764706), # green
    (0.9294117647058824, 0.6941176470588235, 0.12549019607843137), # orange
    (0.49411764705882355, 0.1843137254901961, 0.5568627450980392), # purple
    (0.30196078431372547, 0.7450980392156863, 0.9333333333333333), # cyan
    (0.8823529411764706, 0.43529411764705883, 0.19215686274509805),# brown
    (0.6000000000000000, 0.6000000000000000, 0.6000000000000000), # gray
]

# other graphics constants
START = 'start'
STOP = 'stop'
_DEF_UNITS = 'phy'

ALLOWED_LINESTYLES = {'solid', 'dashed', 'dotted'}
ALLOWED_MARKER_SHAPES = {'sphere', 'cube', 'cone', 'arrow', 'cylinder', 'point'}
ALLOWED_INTERP_TYPES = {'Linear', 'Spline', 'Boolean'}


INT_GRAPHIC_NAMES = {SCENE_KEY}
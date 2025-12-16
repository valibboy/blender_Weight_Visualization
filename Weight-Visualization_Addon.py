bl_info = {
    "name": "Weight Numbers Visualization (Turbo Fixed)",
    "author": "gemini,copilot (adapted)",
    "version": (1, 8, 1),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > Tool",
    "description": "Fully lag-free weight display using NumPy and a time-based throttle. Persistent settings.",
    "warning": "",
    "category": "3D View",
}

import bpy
import blf
import numpy as np
import time
from bpy.app.handlers import persistent

# --- Settings ---
DEFAULT_FONT_SIZE = 16
SCREEN_BUCKET = 14  # Pixel bucket size for overlap suppression

# Global variables
draw_handle = None
last_cache_update_time = 0.0

# Global cache for performance (NumPy optimized)
viz_cache = {
    'weights': None,         # Numpy array of weights
    'coords': None,          # Numpy array of world-space coords (N, 3)
    'normals': None,         # Numpy array of world-space normals (N, 3)
    'indices': None,         # Original vertex indices
    'active_vg_name': None,
    'object_name': None,
    'mesh_token': None,      # Token to detect mesh edits (painting)
    'matrix_world': None
}

# --- Helper Functions ---

def redraw_3d_views():
    """Forces a redraw on all visible 3D Viewports."""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def get_depsgraph(context):
    try:
        return context.evaluated_depsgraph_get()
    except Exception:
        return bpy.context.evaluated_depsgraph_get()

def cache_data(context):
    """
    Refreshes the data cache. Heavy operation.
    """
    global viz_cache

    obj = context.active_object
    if not obj or obj.type != 'MESH':
        viz_cache['coords'] = None
        viz_cache['weights'] = None
        return

    vg = obj.vertex_groups.active
    if not vg:
        viz_cache['coords'] = None
        viz_cache['weights'] = None
        viz_cache['active_vg_name'] = None
        return

    depsgraph = get_depsgraph(context)
    eval_mesh = None
    if depsgraph:
        try:
            obj_eval = obj.evaluated_get(depsgraph)
            eval_mesh = obj_eval.data
        except:
            eval_mesh = obj.data
    
    if not eval_mesh:
        eval_mesh = obj.data

    vg_index = vg.index
    dverts = eval_mesh.vertices
    c_verts = len(dverts)
    
    # --- 1. ROBUST WEIGHT GATHERING (The Bottleneck) ---
    subset_indices = []
    subset_weights = []
    
    for v in dverts:
        for g in v.groups:
            if g.group == vg_index:
                w = g.weight
                if w > 0.001:
                    subset_indices.append(v.index)
                    subset_weights.append(w)
                break
    
    if not subset_indices:
        viz_cache['weights'] = None
        viz_cache['coords'] = None
        viz_cache['active_vg_name'] = vg.name
        mesh_update_tag = getattr(eval_mesh, "update_tag", None)
        viz_cache['mesh_token'] = (id(eval_mesh), len(eval_mesh.vertices), mesh_update_tag)
        return

    # Convert to NumPy
    np_indices = np.array(subset_indices, dtype=np.int32)
    np_weights = np.array(subset_weights, dtype=np.float32)
    
    # --- 2. FAST COORDINATE GATHERING ---
    all_coords = np.zeros(c_verts * 3, dtype=np.float32)
    dverts.foreach_get("co", all_coords)
    all_coords.shape = (c_verts, 3)
    final_coords = all_coords[np_indices]
    
    # --- 3. FAST NORMAL GATHERING ---
    all_normals = np.zeros(c_verts * 3, dtype=np.float32)
    dverts.foreach_get("normal", all_normals)
    all_normals.shape = (c_verts, 3)
    final_normals = all_normals[np_indices]

    # --- 4. APPLY WORLD MATRIX (Vectorized) ---
    matrix_world = obj.matrix_world
    mw = np.array(matrix_world)
    mat3 = mw[:3, :3]
    trans = mw[:3, 3]
    
    final_coords = np.dot(final_coords, mat3.T) + trans
    final_normals = np.dot(final_normals, mat3.T)
    norms = np.linalg.norm(final_normals, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    final_normals /= norms

    # Store in cache
    viz_cache['weights'] = np_weights
    viz_cache['coords'] = final_coords
    viz_cache['normals'] = final_normals
    viz_cache['indices'] = np_indices
    viz_cache['active_vg_name'] = vg.name
    viz_cache['object_name'] = obj.name
    
    mesh_update_tag = getattr(eval_mesh, "update_tag", None)
    viz_cache['mesh_token'] = (id(eval_mesh), len(eval_mesh.vertices), mesh_update_tag)
    viz_cache['matrix_world'] = matrix_world.copy()

# --- Custom Scene Properties ---
def register_scene_props():
    # Note: Defaults here are for new scenes. Persistent settings are applied via Load Post handler.
    bpy.types.Scene.weight_num_color = bpy.props.FloatVectorProperty(
        name="Number Color", subtype='COLOR', default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, size=4
    )
    bpy.types.Scene.weight_num_match_weight_color = bpy.props.BoolProperty(
        name="Match Weight Paint Color", default=False, description="If enabled, numbers use the weight-based color (like weight paint)"
    )
    bpy.types.Scene.weight_num_match_brightness = bpy.props.FloatProperty(
        name="Brightness", default=1.0, min=0.0, max=10.0, precision=3
    )
    bpy.types.Scene.weight_num_fade_threshold = bpy.props.FloatProperty(
        name="Fade Threshold", default=0.0, min=-1.0, max=1.0
    )
    bpy.types.Scene.weight_num_fade_power = bpy.props.FloatProperty(
        name="Fade Curve", default=1.0, min=0.1, max=5.0
    )
    bpy.types.Scene.weight_num_size = bpy.props.IntProperty(
        name="Number Size", default=DEFAULT_FONT_SIZE, min=6, max=96
    )
    bpy.types.Scene.weight_num_decimals = bpy.props.IntProperty(
        name="Decimals", default=2, min=0, max=6
    )
    bpy.types.Scene.weight_num_count_x = bpy.props.IntProperty(
        name="Count X", default=20, min=0, max=4000
    )
    bpy.types.Scene.weight_num_count_y = bpy.props.IntProperty(
        name="Count Y", default=20, min=0, max=4000
    )
    bpy.types.Scene.weight_num_count_size = bpy.props.IntProperty(
        name="Count Size", default=DEFAULT_FONT_SIZE + 4, min=6, max=128
    )
    bpy.types.Scene.weight_num_throttle_ms = bpy.props.IntProperty(
        name="Update Throttle (ms)",
        description="Minimum time (in milliseconds) between heavy cache updates. Higher = smoother viewport during painting.",
        default=150,
        min=10, max=1000,
        subtype='UNSIGNED'
    )

def unregister_scene_props():
    for p in ("weight_num_color", "weight_num_match_weight_color", "weight_num_match_brightness",
              "weight_num_fade_threshold", "weight_num_fade_power", "weight_num_size", "weight_num_decimals",
              "weight_num_count_x", "weight_num_count_y", "weight_num_count_size", "weight_num_throttle_ms"):
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)

# --- PREFERENCES: Persistence Logic ---
BL_ID = __name__

def get_addon_prefs():
    try:
        return bpy.context.preferences.addons[BL_ID].preferences
    except Exception:
        return None

def apply_prefs_to_scene(prefs):
    """Syncs settings from Addon Preferences to the current Scene."""
    if not prefs: return
    scene = getattr(bpy.context, "scene", None)
    if not scene: return
    
    # List of properties to sync
    props = [
        "weight_num_color", "weight_num_match_weight_color", "weight_num_match_brightness",
        "weight_num_fade_threshold", "weight_num_fade_power", "weight_num_size", "weight_num_decimals",
        "weight_num_count_x", "weight_num_count_y", "weight_num_count_size", "weight_num_throttle_ms"
    ]
    
    for attr in props:
        try:
            # Only update if they differ to avoid unnecessary updates
            val = getattr(prefs, attr)
            if getattr(scene, attr) != val:
                setattr(scene, attr, val)
        except: 
            pass

class WeightNum_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = BL_ID

    # These defaults act as the "Master" defaults for the addon
    weight_num_color: bpy.props.FloatVectorProperty(name="Number Color", subtype='COLOR', default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0, size=4)
    weight_num_match_weight_color: bpy.props.BoolProperty(name="Match Weight Paint Color", default=False)
    weight_num_match_brightness: bpy.props.FloatProperty(name="Brightness", default=1.0, min=0.0, max=10.0)
    weight_num_fade_threshold: bpy.props.FloatProperty(name="Fade Threshold", default=0.0, min=-1.0, max=1.0)
    weight_num_fade_power: bpy.props.FloatProperty(name="Fade Curve", default=1.0, min=0.1, max=5.0)
    weight_num_size: bpy.props.IntProperty(name="Number Size", default=DEFAULT_FONT_SIZE, min=6, max=96)
    weight_num_decimals: bpy.props.IntProperty(name="Decimals", default=2, min=0, max=6)
    weight_num_count_x: bpy.props.IntProperty(name="Count X", default=20, min=0, max=4000)
    weight_num_count_y: bpy.props.IntProperty(name="Count Y", default=20, min=0, max=4000)
    weight_num_count_size: bpy.props.IntProperty(name="Count Size", default=DEFAULT_FONT_SIZE + 4, min=6, max=128)
    weight_num_throttle_ms: bpy.props.IntProperty(name="Update Throttle (ms)", default=150, min=10, max=1000)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Default Settings (Saved Globally):")
        layout.prop(self, "weight_num_color")
        layout.prop(self, "weight_num_match_weight_color")
        if self.weight_num_match_weight_color:
            layout.prop(self, "weight_num_match_brightness")
        layout.prop(self, "weight_num_size")
        layout.prop(self, "weight_num_decimals")
        layout.prop(self, "weight_num_count_size")
        layout.prop(self, "weight_num_count_x")
        layout.prop(self, "weight_num_count_y")
        layout.prop(self, "weight_num_fade_threshold")
        layout.prop(self, "weight_num_fade_power")
        layout.prop(self, "weight_num_throttle_ms")

class WM_OT_SaveWeightNumPreferences(bpy.types.Operator):
    bl_idname = "wm.save_weight_num_prefs"
    bl_label = "Save Preferences"
    bl_description = "Save current Sidebar settings as Global Defaults for startup"

    def execute(self, context):
        scene = getattr(context, "scene", None)
        if not scene: return {'CANCELLED'}
        prefs = get_addon_prefs()
        if not prefs: 
            self.report({'ERROR'}, "Could not find Addon Preferences")
            return {'CANCELLED'}

        props = [
            "weight_num_color", "weight_num_match_weight_color", "weight_num_match_brightness",
            "weight_num_fade_threshold", "weight_num_fade_power", "weight_num_size", "weight_num_decimals",
            "weight_num_count_x", "weight_num_count_y", "weight_num_count_size", "weight_num_throttle_ms"
        ]

        # Copy Scene -> Prefs
        for attr in props:
            try:
                setattr(prefs, attr, getattr(scene, attr))
            except: 
                pass
        
        # Save to userpref.blend
        bpy.ops.wm.save_userpref()
        self.report({'INFO'}, "Weight Viz preferences saved!")
        return {'FINISHED'}

@persistent
def load_handler(dummy):
    """Applies saved preferences when a file is loaded."""
    prefs = get_addon_prefs()
    apply_prefs_to_scene(prefs)

# --- Drawing Function (NumPy Turbo with Throttle) ---
def draw_callback_px(self, context):
    global viz_cache, last_cache_update_time

    obj = context.active_object
    if not obj or context.mode != 'PAINT_WEIGHT':
        return
    
    active_vg = obj.vertex_groups.active
    active_vg_name = active_vg.name if active_vg else None

    needs_refresh = False

    # Check 1: Object/Bone Switching
    if obj.name != viz_cache.get('object_name'):
        needs_refresh = True
    elif active_vg_name != viz_cache.get('active_vg_name'):
        needs_refresh = True
    
    # Check 2: Live Painting (Mesh Content Update)
    if not needs_refresh:
        try:
            depsgraph = get_depsgraph(context)
            if depsgraph:
                eval_obj = obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.data
            else:
                eval_mesh = obj.data
        except:
            eval_mesh = obj.data
        
        current_token = (id(eval_mesh), len(eval_mesh.vertices), getattr(eval_mesh, "update_tag", None))
        if viz_cache.get('mesh_token') != current_token:
            needs_refresh = True

    # --- THROTTLE LOGIC ---
    current_time = time.time()
    throttle_ms = context.scene.weight_num_throttle_ms

    if needs_refresh:
        # Check if enough time has passed since the last heavy update
        if (current_time - last_cache_update_time) * 1000.0 >= throttle_ms:
            # Full cache update (slow part)
            cache_data(context)
            last_cache_update_time = current_time
        else:
            # Throttle active: skip the heavy cache update this frame
            pass 

    # If cache is still empty (e.g., Zero Weight Bone), we draw nothing
    coords = viz_cache.get('coords')
    
    region = context.region
    rv3d = context.region_data
    if not rv3d: return

    if coords is None or len(coords) == 0:
        # If cache is empty, we still draw the count (which will be 0)
        width = region.width
        height = region.height
        cx = max(0, min(context.scene.weight_num_count_x, width))
        cy = max(0, min(context.scene.weight_num_count_y, height))
        c_size = context.scene.weight_num_count_size
        
        font_id = 0
        blf.size(font_id, c_size)
        blf.color(font_id, 1, 1, 1, 0.95)
        blf.position(font_id, cx, cy, 0)
        blf.draw(font_id, f"Visible Weighted Vertices: 0")
        return

    # --- NumPy Calculations (FAST) ---
    N = len(coords)
    
    # Matrices
    persp_matrix = rv3d.perspective_matrix
    pm = np.array(persp_matrix)
    view_inv = np.array(rv3d.view_matrix.inverted())
    cam_loc = view_inv[:3, 3]
    
    # Projection
    ones = np.ones((N, 1), dtype=np.float32)
    coords_4d = np.hstack((coords, ones))
    clip_coords = np.dot(coords_4d, pm.T)
    
    x, y, z, w = clip_coords[:,0], clip_coords[:,1], clip_coords[:,2], clip_coords[:,3]
    
    # Filter 1: Frustum Culling
    mask = (w > 0.001) & (np.abs(x) <= w * 1.1) & (np.abs(y) <= w * 1.1)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0: return
        
    # Subset Data
    s_coords = coords[valid_indices]
    s_normals = viz_cache['normals'][valid_indices]
    s_weights = viz_cache['weights'][valid_indices]
    s_clip = clip_coords[valid_indices]
    
    # Filter 2: Backface Culling & Fading
    cam_vecs = cam_loc - s_coords
    dots = np.einsum('ij,ij->i', s_normals, cam_vecs)
    dist_sq = np.einsum('ij,ij->i', cam_vecs, cam_vecs)
    dists = np.sqrt(dist_sq)
    dists[dists < 1e-6] = 1.0
    facings = dots / dists
    
    threshold = context.scene.weight_num_fade_threshold
    mask_face = facings > threshold
    
    valid_indices_2 = np.where(mask_face)[0]
    if len(valid_indices_2) == 0: return
        
    f_weights = s_weights[valid_indices_2]
    f_facings = facings[valid_indices_2]
    f_clip = s_clip[valid_indices_2]
    f_dists = dists[valid_indices_2]
    
    # Map to Screen
    f_w = f_clip[:, 3]
    ndc_x = f_clip[:, 0] / f_w
    ndc_y = f_clip[:, 1] / f_w
    
    width = region.width
    height = region.height
    screen_x = (ndc_x + 1.0) * 0.5 * width
    screen_y = (ndc_y + 1.0) * 0.5 * height
    
    # --- Bucketing & Drawing (The Python Loop) ---
    buckets = {}
    bucket_size = SCREEN_BUCKET
    fade_power = context.scene.weight_num_fade_power
    user_alpha = context.scene.weight_num_color[3]
    
    sx_list = screen_x.tolist()
    sy_list = screen_y.tolist()
    w_list = f_weights.tolist()
    face_list = f_facings.tolist()
    dist_list = f_dists.tolist()
    
    denom = (1.0 - threshold)
    if denom == 0.0: denom = 1.0
    
    for i in range(len(sx_list)):
        px, py = sx_list[i], sy_list[i]
        if not (0 <= px < width and 0 <= py < height): continue
            
        bx, by = int(px // bucket_size), int(py // bucket_size)
        key = (bx, by)
        depth = dist_list[i]
        
        if key in buckets and depth >= buckets[key][0]: continue
            
        facing = face_list[i]
        t = max(0.0, min(1.0, (facing - threshold) / denom))
        alpha = user_alpha * (t ** fade_power)
        
        if alpha < 0.05: continue
        buckets[key] = (depth, px, py, w_list[i], alpha)

    # Actual Drawing
    font_id = 0
    font_size = context.scene.weight_num_size
    blf.size(font_id, font_size)
    
    match_color = context.scene.weight_num_match_weight_color
    u_col = context.scene.weight_num_color
    bright = context.scene.weight_num_match_brightness
    decimals = context.scene.weight_num_decimals
    fmt = "{:." + str(decimals) + "f}"
    
    count = 0
    for depth, px, py, w, alpha in buckets.values():
        if match_color:
            wv = max(0.0, min(1.0, w))
            if wv <= 0.5:
                t = wv * 2.0
                r, g, b = 0.0, t, 1.0 - t
            else:
                t = (wv - 0.5) * 2.0
                r, g, b = t, 1.0 - t, 0.0
            blf.color(font_id, r*bright, g*bright, b*bright, alpha)
        else:
            blf.color(font_id, u_col[0], u_col[1], u_col[2], alpha)
            
        blf.position(font_id, px, py, 0)
        blf.draw(font_id, fmt.format(w))
        count += 1
        
    # Draw Stats
    cx = max(0, min(context.scene.weight_num_count_x, width))
    cy = max(0, min(context.scene.weight_num_count_y, height))
    c_size = context.scene.weight_num_count_size
    
    blf.size(font_id, c_size)
    blf.color(font_id, 1, 1, 1, 0.95)
    blf.position(font_id, cx, cy, 0)
    blf.draw(font_id, f"Visible Weighted Vertices: {count}")

# --- Operators & Panels ---

class WM_OT_ToggleWeightNumbers(bpy.types.Operator):
    bl_idname = "view3d.toggle_weight_numbers"
    bl_label = "Toggle Weight Numbers"

    def execute(self, context):
        global draw_handle, last_cache_update_time
        if draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(draw_handle, 'WINDOW')
            draw_handle = None
            viz_cache['coords'] = None
            viz_cache['weights'] = None
            redraw_3d_views()
        else:
            cache_data(context)
            last_cache_update_time = time.time() # Reset timer on activation
            draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
            redraw_3d_views()
        return {'FINISHED'}

class VIEW3D_PT_WeightNumPanel(bpy.types.Panel):
    bl_label = "Weight Visualization"
    bl_idname = "VIEW3D_PT_weight_viz"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        if draw_handle:
            layout.operator("view3d.toggle_weight_numbers", text="Stop Visualization", icon='CANCEL')
            layout.prop(scene, "weight_num_color")
            layout.prop(scene, "weight_num_match_weight_color")
            if scene.weight_num_match_weight_color:
                layout.prop(scene, "weight_num_match_brightness")
            layout.prop(scene, "weight_num_size")
            layout.prop(scene, "weight_num_decimals")
            layout.prop(scene, "weight_num_count_size")
            layout.prop(scene, "weight_num_count_x")
            layout.prop(scene, "weight_num_count_y")
            layout.separator()
            layout.prop(scene, "weight_num_throttle_ms")
            layout.separator()
            layout.prop(scene, "weight_num_fade_threshold")
            layout.prop(scene, "weight_num_fade_power")
            layout.separator()
            layout.operator("wm.save_weight_num_prefs", icon='FILE_TICK')
        else:
            layout.operator("view3d.toggle_weight_numbers", text="Show Numbers", icon='PLAY')

class PROPERTIES_PT_WeightNumPanel(bpy.types.Panel):
    bl_label = "Weight Visualization"
    bl_idname = "PROPERTIES_PT_weight_viz"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    def draw(self, context):
        VIEW3D_PT_WeightNumPanel.draw(self, context)

# --- Register ---
classes = (
    WM_OT_ToggleWeightNumbers,
    VIEW3D_PT_WeightNumPanel,
    PROPERTIES_PT_WeightNumPanel,
    WeightNum_AddonPreferences,
    WM_OT_SaveWeightNumPreferences,
)

def register():
    register_scene_props()
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.app.handlers.load_post.append(load_handler)

def unregister():
    global draw_handle
    if draw_handle:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handle, 'WINDOW')
        draw_handle = None
        
    bpy.app.handlers.load_post.remove(load_handler)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_scene_props()

if __name__ == "__main__":
    register()
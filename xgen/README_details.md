# XGen

XGen is a 3D HOI data visualization and synthesis tool built with MuJoCo and
Three.js. It uses the G1 MuJoCo model and meshes to display robot motion and
supports:

- Browsing robot-only and complete HOI `.npz` data;
- Frame-by-frame inspection, timeline scrubbing, and playback;
- 29-DoF joint editing;
- Editing object position, orientation, geometry, and size;
- Box and sphere objects;
- Contact start/end markers;
- Static, catch-parabola, and throw-parabola trajectory synthesis;
- A draggable initial-velocity arrow for parabolic trajectories;
- MuJoCo optimization of the hands and object during the contact stage;
- Cropping a time range and exporting a new `.npz` file;
- Converting legacy pickle motion data into XGen robot-only data.

For a video-to-HOI workflow, use:

- [GVHMR](https://github.com/zju3dv/GVHMR): video to human motion;
- [UMR](https://github.com/hanyang9/UMR): human motion to robot motion;
- XGen: robot motion to HOI data.

## Directory Structure

From the HumanX repository root, XGen is located under `xgen/`. The commands
below assume that you have entered the `xgen` directory:

```bash
cd xgen
```

The files required by the released XGen workflow are organized as follows:

```text
xgen/
├── data/
│   ├── example/             # Bundled example NPZ files
│   └── <dataset>/           # Local NPZ batches, such as speed_capped_5mps/
└── xgen_editor/
    ├── hoi_editor/
    │   ├── server.py        # Local server and MuJoCo backend
    │   ├── app.js           # 3D frontend
    │   ├── styles.css
    │   ├── index.html
    │   ├── convert_motion_pkl.py
    │   └── convert_g1_motion.py
    ├── decoupled_wbc/
    │   └── control/robot_model/model_data/g1/
    │       ├── g1_29dof_old.xml
    │       └── meshes/
    └── run_hoi_editor.sh
```

`decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml` and its
`meshes/` directory are required XGen resources. Do not delete or move them.

## Requirements

- Python 3;
- A desktop browser with WebGL support;
- The following Python packages:

```bash
python -m pip install numpy mujoco
```

To convert `.pkl` files, also install:

```bash
python -m pip install joblib
```

We recommend running XGen in a Python environment where MuJoCo is already
installed. The Three.js files used by the XGen frontend are vendored under
`hoi_editor/vendor/`, so no Node.js packages or network access are required.

## Launch XGen

From the `xgen` directory, run:

```bash
bash xgen_editor/run_hoi_editor.sh
```

Then open:

```text
http://127.0.0.1:8765
```

You can also start the Python server directly:

```bash
python xgen_editor/hoi_editor/server.py --port 8765
```

If the port is already in use, choose another port:

```bash
bash xgen_editor/run_hoi_editor.sh --port 8766
```

The default data root is `xgen/data/`. You can specify a different root:

```bash
python xgen_editor/hoi_editor/server.py \
  --data-root /path/to/data \
  --port 8765
```

## Ground Height and Color

The visualizer uses the minimum world-space `z` coordinate of the eight
left- and right-foot contact sites in the first loaded frame as the ground
reference. These sites correspond to `left_foot_contact_*` and
`right_foot_contact_*` in the G1 MuJoCo XML:

```text
ground_z = min(all_left_and_right_foot_contact_site_pos_w.z)
```

In the current schema, entries `2` and `3` of `ee_pos_w` are the left and
right ankle-roll bodies. `ee_quat_w` is used to transform the local foot
contact-site coordinates from the XML into world coordinates. The resulting
ground height remains fixed while the current file is being browsed and is
translated to `z=0` in the visualization. As a result, feet above this plane
appear to float, while feet below it appear to penetrate the ground.

The `Ground` button at the top of the 3D view cycles through several
high-contrast ground colors. It only changes the Three.js visualization; it
does not modify or save the NPZ data and does not affect the robot or object
trajectories.

The toolbar also includes a `Grounding correction` switch, enabled by default.
When enabled, XGen uses foot contact-site heights and foot velocity to identify
support phases and smoothly corrects the robot's vertical display position.
The correction is frozen when both feet are simultaneously off the ground and
smoothly resumes when the robot lands. With the switch disabled, the original
data is saved. With it enabled, saving, cropping, and synthesis write the same
per-frame `z` correction to the exported data and recompute FK and velocity.
Objects and supporting surfaces in HOI data are translated together to preserve
the displayed contact relationship. Exported files record
`grounding_correction_applied`, `grounding_correction_z`, and
`grounding_ground_z`.

## Data Types

XGen recursively browses `.npz` files under the current data root. The default
data root is `xgen/data/`. A robot-only file must contain at least:

```text
fps
base_pos_w
base_quat_w
joint_pos
```

The recommended complete robot-only schema is:

```text
fps                  (1,)
base_pos_w           (T, 3)
base_quat_w          (T, 4)       # wxyz
joint_pos            (T, 29)
joint_vel            (T, 29)
body_pos_w           (T, 41, 3)
body_quat_w          (T, 41, 4)
body_lin_vel_w       (T, 41, 3)
body_ang_vel_w       (T, 41, 3)
ee_pos_w             (T, 5, 3)
ee_quat_w            (T, 5, 4)
```

When the following fields are present, XGen recognizes the file as complete
HOI data:

```text
object_pos_w
object_quat_w
```

Complete HOI data commonly also contains:

```text
object_lin_vel_w
object_ang_vel_w
object_size
object_type             # "box" or "sphere"
contact_info
support_enabled
support_size
table1_pos_w
table2_pos_w
```

Legacy data containing `support1_pos_w` and `support2_pos_w` can also be read.
Newly exported files use `table1_pos_w` and `table2_pos_w`.

The meaning of `object_size` depends on `object_type`:

- `box`: edge length of the box;
- `sphere`: diameter of the sphere.

Quaternions in XGen data consistently use `wxyz` order.

## Basic Editing

### Dataset Folders

Select a batch directory from the `Dataset folder` dropdown under
`HOI datasets` on the left. `New folder` creates a subdirectory in the current
directory and switches to it automatically. After selecting a directory, the
file list only shows `.npz` files in that directory. Outputs from `Save copy`,
`Crop copy`, and `Save synthesized HOI` are written directly to the selected
folder.

### Robot and Joints

1. Select a data file from the file list on the left;
2. Use sliders or numeric inputs in the `Joints` tab to edit joints;
3. You can also drag joint markers directly in the 3D scene;
4. Click `Save copy` at the top to save an edited copy.

Editing does not overwrite the source file. Edited copies are saved to the
currently selected dataset folder.

### Object Configuration

The `Object config` tab allows you to edit:

- `Box` or `Sphere`;
- Object position;
- Object quaternion;
- Object rotation;
- Whether to keep the object level and follow yaw only;
- Whether to generate supporting surfaces;
- Object size.

There are two object-size modes in `Size mode`:

- `Auto · wrist span`: for robot-only data, use the current distance between
  the two wrists; for existing HOI data, prefer the `object_size` saved in the
  file;
- `Manual`: use the size specified in the input field. For a box this is the
  edge length; for a sphere this is the diameter.

Manual sizes are specified in meters. The interface displays the geometry as:

- `Box edge length`: edge length of the box;
- `Sphere diameter`: diameter of the sphere.

User changes to the object type, size mode and size, supporting-surface
switch, level-yaw lock, and pre/post-contact trajectory modes are saved as
browser-local XGen batch-editing preferences. These options are reused when
switching to another file. Contact frames, object anchors, velocity vectors,
and optimization previews are still reset from the current data.

## Timeline and Cropping

The timeline supports three interactions:

- Click or drag the timeline to change the current frame;
- Drag the left handle to set the crop start frame;
- Drag the right handle to set the crop end frame.

Click `Crop copy`, enter an output filename, and save a cropped copy. All
frame-major arrays are sliced consistently, and contact/release frame indices
are recomputed.

Cropped copies are saved to the currently selected dataset folder. This makes
it possible to keep different data batches in separate directories.

For an existing complete HOI file, use `Crop selected HOI` at the bottom of
the Synthesis tab directly; contact markers do not need to be set again.

## HOI Synthesis Workflow

HOI synthesis takes a robot-only `.npz` file as input.

### 1. Mark the Contact Stage

1. Select a robot-only file;
2. Open the `Synthesis` tab;
3. Move the timeline to the contact start frame and click `Mark start`;
4. Move the timeline to the contact end frame and click `Mark end`.

The current workflow assumes one continuous contact stage.

### 2. Set the Object Anchor and Size

When you click `Align object`, the object anchor is aligned to the center of
the two palm centers, rather than the wrist joint centers. The default object
size is the distance between the two wrists.

You can then use `Object config` to:

- Select `Box` or `Sphere`;
- Enter a box edge length or sphere diameter manually;
- Adjust object position and rotation;
- Enable or disable the level-yaw lock;
- Enable or disable supporting surfaces.

During the contact stage, the relative pose between the object and both palm
centers remains fixed.

### 3. Set the Pre-Contact Trajectory

`Pre-contact trajectory` has two modes:

- `Hold still`: keep the object stationary before contact starts;
- `Inverted parabolic`: compute a reverse parabola from the object position at
  contact start so that the object reaches the target position exactly at
  contact start.

When browsing the pre-contact period, the interface displays the pre-contact
parabola and velocity arrow. The velocity arrow can be dragged in the 3D scene.

### 4. Set the Post-Contact Trajectory

`Post-contact trajectory` has two modes:

- `Hold still`: keep the object at the contact-end pose after contact;
- `Parabolic throw`: move the object along a parabola after contact ends.

By default, the post-contact initial velocity is computed from the average
wrist-center velocity over the five frames following contact end and uses
Earth gravity:

```text
gravity = [0, 0, -9.80665] m/s^2
```

The displayed velocity arrow is `1/5` of the actual velocity so that it is
easy to drag. Dragging the arrow immediately updates the parabola preview and
export parameters. Pre- and post-contact parabolas are hidden during the
contact stage.

### 5. Contact Optimization

Click `Optimize contact stage` to run optimization and preview the result
before exporting.

The current optimization:

- Optimizes only the robot arm joints;
- Keeps the object position, orientation, and trajectory fixed;
- Brings both palms closer to the object;
- Considers palm orientation, object-surface distance, hand penetration, and
  two-hand contact balance;
- Smooths the transitions before and after contact to avoid visible jumps;
- Supports both `Box` and `Sphere`;
- Uses the saved sphere diameter for sphere contact calculations.

The optimization button is only available for the robot-only synthesis
workflow. Existing complete HOI data cannot be re-optimized with this button,
but it can still be browsed, edited, and cropped.

Click `Restore original` to remove the optimization preview.

### 6. Export

After clicking `Save synthesized HOI`, XGen generates a complete HOI `.npz`
file in the currently selected dataset folder.

The file contains robot trajectories, object trajectories, contact
information, object type and size, as well as parabolic-trajectory and
optimization metadata.

## Example: Browse Existing HOI Data

The public tree includes the following examples under `data/example/`,
which can be opened directly.

After starting the editor, select `example` under `HOI datasets` on the left
and then select the `.npz` file to inspect the robot, object, contact,
trajectory, and grounding information.

## Example: Synthesize Catch-and-Throw HOI Data

If you have a robot-only motion such as
`data/robot_only/catch_middle-hold-throw_robot_only.npz`, follow the workflow
below to synthesize a "catch the ball and throw it again" HOI motion.
`data/robot_only/` and `data/synthesized/` are recommended local working
directories and do not need to be committed with the source code.

1. **Select the robot data**

   Select your robot-only `.npz` file from the file list on the left.

2. **Configure the sphere**

   Open `Object config`, select `Sphere`, and set the sphere diameter. The
   default size is computed from the current motion, but you can also enter a
   manual size and adjust the object position, rotation, and other settings.

3. **Mark the contact stage**

   Open `Synthesis`, move the timeline to the frame where both hands begin to
   contact the ball, and click `Mark start`. Move to the frame where contact
   ends and click `Mark end`. The current workflow supports one continuous
   contact stage.

4. **Set the incoming and outgoing trajectories**

   Set `Pre-contact trajectory` to `Inverted parabolic` and
   `Post-contact trajectory` to `Parabolic throw`.

   - The pre-contact parabola is computed in reverse so that the ball reaches
     the configured position exactly at `contact start`, simulating an
     incoming ball;
   - The post-contact parabola starts at `contact end` and simulates the ball
     being thrown again;
   - When the current frame is in the pre- or post-contact stage, the 3D scene
     displays the corresponding dashed trajectory and velocity vector;
   - Drag the velocity vector for the current stage to adjust the parabola
     direction and initial velocity. Neither trajectory is displayed during
     contact;
   - Default gravity is `-9.80665 m/s^2`, and the displayed velocity arrow is
     `1/5` of the actual velocity.

5. **Optimize contact**

   Click `Optimize contact stage`, wait for optimization to finish, and
   inspect the result in the 3D view. Optimization only adjusts the robot arm
   joints so that both palms stay stably attached to the ball while reducing
   penetration, floating, and transition jumps. The ball trajectory remains
   unchanged. Click `Restore original` to restore the robot motion before
   optimization.

6. **Save the synthesized data**

   Click `Save synthesized HOI`, enter a filename, and save. The complete HOI
   data is written to the currently selected dataset folder.

   The exported file stores robot and ball trajectories, contact information,
   ball type and diameter, pre/post-contact parabola parameters, and contact
   optimization metadata.

7. **Crop the time range**

   Use the two timeline handles to select the desired start and end frames,
   then click `Crop selected HOI`. You can also select the exported complete
   HOI file from the left file list and crop it from the `Synthesis` tab
   without marking contact again.

   The cropped file is written directly to the currently selected dataset
   folder. All frame-major arrays are cropped consistently, and the contact
   start/end frame indices are recomputed.

## Convert Legacy Pickle Data

For legacy pickle files containing:

```text
root_trans_offset
root_rot
dof
obj_pos
obj_rot_quat
```

convert them to complete HOI format with:

```bash
python xgen_editor/hoi_editor/convert_motion_pkl.py \
  data/input.pkl \
  data/output.npz
```

To generate robot-only data, use `--robot-only`:

```bash
python xgen_editor/hoi_editor/convert_motion_pkl.py \
  data/input.pkl \
  data/robot_only/input_robot_only.npz \
  --robot-only
```

The converter:

- Reads nested or non-nested pickle dictionaries;
- Converts `root_rot` and `obj_rot_quat` from `xyzw` to `wxyz`;
- Maps the 29-DoF joints to the XGen joint order;
- Recomputes body and end-effector FK with MuJoCo;
- Recomputes velocity fields.

`--robot-only` mode does not write `object_*`, `contact_info`, or supporting
surface fields.

## Convert g1_motion NPZ Data

For an input g1 motion NPZ supported by the XGen converter, the default fields
include:

```text
root_pos
root_quat_wxyz
joint_pos
joint_names
```

Run:

```bash
python xgen_editor/hoi_editor/convert_g1_motion.py \
  data/g1_motion \
  --output-dir data/robot_only
```

The converter reorders joints according to `joint_names` and recomputes MuJoCo
FK. Output filenames automatically receive the `_robot_only.npz` suffix.

## Troubleshooting

### A file does not appear in the file list

Check that the file has a `.npz` extension and contains at least:

```text
base_pos_w
base_quat_w
joint_pos
```

Also verify that the server's `--data-root` points to the correct `data/`
directory.

### The 3D scene is empty or the browser reports a WebGL error

Use a desktop browser with WebGL support. If hardware acceleration or the
graphics driver is disabled, Three.js cannot create a rendering context.

### The G1 model or meshes cannot be found

Verify that the following paths exist:

```text
xgen_editor/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_old.xml
xgen_editor/decoupled_wbc/control/robot_model/model_data/g1/meshes/
```

### The port is already in use

Start the server on another port, for example:

```bash
bash xgen_editor/run_hoi_editor.sh --port 8766
```

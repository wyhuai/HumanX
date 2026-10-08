
class Viewer:
    """This class handles controlling the camera associated with a viewport in the simulator.

    It can be used to set the viewpoint camera to track different origin types:

    - **world**: the center of the world (static)
    - **env**: the center of an environment (static)
    - **asset_root**: the root of an asset in the scene (e.g. tracking a robot moving in the scene)

    On creation, the camera is set to track the origin type specified in the configuration.

    For the :attr:`asset_root` origin type, the camera is updated at each rendering step to track the asset's
    root position. For this, it registers a callback to the post update event stream from the simulation app.
    """

    def __init__(self):
        """Initialize the ViewportCameraController.

        Args:
            env: The environment.
            cfg: The configuration for the viewport camera controller.

        Raises:
            ValueError: If origin type is configured to be "env" but :attr:`cfg.env_index` is out of bounds.
            ValueError: If origin type is configured to be "asset_root" but :attr:`cfg.asset_name` is unset.

        """
        self._cfg = None
        return

    def __del__(self):
        """Unsubscribe from the callback."""
        # use hasattr to handle case where __init__ has not completed before __del__ is called
        return

    """
    Properties
    """

    @property
    def cfg(self):
        """The configuration for the viewer."""
        return self._cfg

    """
    Public Functions
    """

    def set_view_env_index(self, env_index: int):
        """Sets the environment index for the camera view.

        Args:
            env_index: The index of the environment to set the camera view to.

        Raises:
            ValueError: If the environment index is out of bounds. It should be between 0 and num_envs - 1.
        """
        return

    def update_view_to_world(self):
        """Updates the viewer's origin to the origin of the world which is (0, 0, 0)."""
        return

    def update_view_to_env(self):
        """Updates the viewer's origin to the origin of the selected environment."""
        return

    def update_view_to_asset_root(self, asset_name: str):
        """Updates the viewer's origin based upon the root of an asset in the scene.

        Args:
            asset_name: The name of the asset in the scene. The name should match the name of the
                asset in the scene.

        Raises:
            ValueError: If the asset is not in the scene.
        """
        return

    def update_view_location(self, eye, lookat):
        """Updates the camera view pose based on the current viewer origin and the eye and lookat positions.

        Args:
            eye: The eye position of the camera. If None, the current eye position is used.
            lookat: The lookat position of the camera. If None, the current lookat position is used.
        """
        return

    """
    Private Functions
    """

    def _update_tracking_callback(self, event):
        """Updates the camera view at each rendering step."""
        return

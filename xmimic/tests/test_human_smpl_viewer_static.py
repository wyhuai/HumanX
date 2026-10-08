import unittest
from html.parser import HTMLParser
from pathlib import Path


class _ElementParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.by_id = {}

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        node = {"tag": tag, "attrs": attr_map, "parents": list(self.stack)}
        if "id" in attr_map:
            self.by_id[attr_map["id"]] = node
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.stack:
            while self.stack:
                popped = self.stack.pop()
                if popped == tag:
                    break


class HumanSmplViewerStaticTests(unittest.TestCase):
    def test_frame_controls_are_independent_and_have_button_fallbacks(self):
        html = Path("tools/human_smpl_viewer/static/index.html").read_text()
        parser = _ElementParser()
        parser.feed(html)

        frame_slider = parser.by_id["frameSlider"]
        viewport_start = html.find('<section class="viewport-panel">')
        viewport_end = html.find("</section>", viewport_start)
        timeline_start = html.find('id="timelinePanel"')
        self.assertNotIn("label", frame_slider["parents"])
        self.assertGreater(timeline_start, viewport_start)
        self.assertLess(timeline_start, viewport_end)
        self.assertIn("prevFrameButton", parser.by_id)
        self.assertIn("nextFrameButton", parser.by_id)
        self.assertIn("frameNumber", parser.by_id)
        self.assertIn("orientationToggle", parser.by_id)
        self.assertIn("curveModeSelect", parser.by_id)

    def test_initial_camera_fits_current_frame_not_whole_motion(self):
        js = Path("tools/human_smpl_viewer/static/app.js").read_text()

        self.assertIn("viewOffset: new THREE.Vector3()", js)
        self.assertIn("state.viewOffset.copy(computeFrameCenter(state.frame))", js)
        self.assertIn("return rawToThreeBase(point).sub(state.viewOffset)", js)
        self.assertIn("fitCameraToFrame(state.frame)", js)
        self.assertIn("function fitCameraToFrame(frameIndex)", js)
        self.assertNotIn("function fitCameraToMotion()", js)

    def test_orientation_visualization_is_wired_to_payload(self):
        js = Path("tools/human_smpl_viewer/static/app.js").read_text()

        self.assertIn("rootOrientArrow", js)
        self.assertIn("human_global_orient_quat", js)
        self.assertIn("human_global_orient_quat_xyzw", js)
        self.assertIn("updateRootOrientation()", js)
        self.assertIn('curveMode === "orient"', js)
        self.assertIn("human_global_orient_quat raw WXYZ", js)


if __name__ == "__main__":
    unittest.main()

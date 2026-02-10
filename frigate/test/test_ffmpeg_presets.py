import unittest
from types import SimpleNamespace
from unittest.mock import patch

from frigate.config import FrigateConfig
from frigate.config.camera.ffmpeg import FFMPEG_INPUT_ARGS_DEFAULT
from frigate.ffmpeg_presets import (
    _gpu_selector,
    parse_preset_hardware_acceleration_decode,
    parse_preset_hardware_acceleration_scale,
    parse_preset_input,
)


class TestFfmpegPresets(unittest.TestCase):
    def setUp(self):
        self.default_ffmpeg = {
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "back": {
                    "ffmpeg": {
                        "inputs": [
                            {
                                "path": "rtsp://10.0.0.1:554/video",
                                "roles": ["detect"],
                            }
                        ],
                        "output_args": {
                            "detect": "-f rawvideo -pix_fmt yuv420p",
                            "record": "-f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1 -strftime 1 -c copy -an",
                        },
                    },
                    "detect": {
                        "height": 1080,
                        "width": 1920,
                        "fps": 5,
                    },
                    "record": {
                        "enabled": True,
                    },
                    "name": "back",
                }
            },
        }

    def test_default_ffmpeg(self):
        FrigateConfig(**self.default_ffmpeg)

    def test_ffmpeg_hwaccel_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "preset-rpi-64-h264"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-rpi-64-h264" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert "-c:v:1 h264_v4l2m2m" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "-other-hwaccel args"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-other-hwaccel args" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_scale_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "preset-nvidia-h264"
        )
        self.default_ffmpeg["cameras"]["back"]["detect"] = {
            "height": 1920,
            "width": 2560,
            "fps": 10,
        }
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-nvidia-h264" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert (
            "fps=10,scale_cuda=w=2560:h=1920,hwdownload,format=nv12,eq=gamma=1.4:gamma_weight=0.5"
            in (" ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"]))
        )

    def test_default_ffmpeg_input_arg_preset(self):
        frigate_config = FrigateConfig(**self.default_ffmpeg)

        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = (
            "preset-rtsp-generic"
        )
        frigate_preset_config = FrigateConfig(**self.default_ffmpeg)
        assert (
            # Ignore global and user_agent args in comparison
            frigate_preset_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
            == frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
        )

    def test_ffmpeg_input_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = (
            "preset-rtmp-generic"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-rtmp-generic" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert (" ".join(parse_preset_input("preset-rtmp-generic", 5))) in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_input_args_as_string(self):
        # Strip user_agent args here to avoid handling quoting issues
        defaultArgsList = parse_preset_input(FFMPEG_INPUT_ARGS_DEFAULT, 5)[2::]
        argsString = " ".join(defaultArgsList) + ' -some "arg with space"'
        argsList = defaultArgsList + ["-some", "arg with space"]
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = argsString
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert set(argsList).issubset(
            frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"]
        )

    def test_ffmpeg_input_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["input_args"] = "-some inputs"
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-some inputs" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_output_record_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["output_args"]["record"] = (
            "preset-record-generic-audio-aac"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "preset-record-generic-audio-aac" not in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )
        assert "-c:v copy -c:a aac" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_output_record_not_preset(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["output_args"]["record"] = (
            "-some output -segment_time 10"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-some output" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_device_override(self):
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_args"] = (
            "preset-intel-qsv-h264"
        )
        self.default_ffmpeg["cameras"]["back"]["ffmpeg"]["hwaccel_device"] = (
            "/dev/dri/renderD129"
        )
        frigate_config = FrigateConfig(**self.default_ffmpeg)
        assert "-qsv_device /dev/dri/renderD129" in (
            " ".join(frigate_config.cameras["back"].ffmpeg_cmds[0]["cmd"])
        )

    def test_ffmpeg_hwaccel_gpu_index_out_of_range_uses_first_device(self):
        _gpu_selector._valid_gpus = ["/dev/dri/renderD128"]
        try:
            args = parse_preset_hardware_acceleration_decode(
                "preset-intel-qsv-h264",
                5,
                1920,
                1080,
                1,
            )
            assert "/dev/dri/renderD128" in args
        finally:
            _gpu_selector._valid_gpus = None

    def test_ffmpeg_hwaccel_gpu_devices_are_sorted(self):
        _gpu_selector._valid_gpus = None
        with patch("frigate.ffmpeg_presets.os.path.exists", return_value=True), patch(
            "frigate.ffmpeg_presets.os.listdir",
            return_value=["renderD129", "renderD128"],
        ), patch(
            "frigate.ffmpeg_presets.vainfo_hwaccel",
            return_value=SimpleNamespace(returncode=0),
        ):
            args = parse_preset_hardware_acceleration_decode(
                "preset-intel-qsv-h264",
                5,
                1920,
                1080,
                0,
            )
            assert "/dev/dri/renderD128" in args
        _gpu_selector._valid_gpus = None

    def test_ffmpeg_qsv_falls_back_to_vaapi_for_xe_driver(self):
        with patch.object(_gpu_selector, "is_xe_driver", return_value=True):
            args = parse_preset_hardware_acceleration_decode(
                "preset-intel-qsv-h264",
                5,
                1920,
                1080,
                0,
                "/dev/dri/renderD128",
            )
            args_str = " ".join(args)
            assert "-hwaccel vaapi" in args_str
            assert "-hwaccel_device /dev/dri/renderD128" in args_str

    def test_ffmpeg_qsv_scale_falls_back_to_vaapi_for_xe_driver(self):
        with patch.object(_gpu_selector, "is_xe_driver", return_value=True):
            args = parse_preset_hardware_acceleration_scale(
                "preset-intel-qsv-h264",
                ["-f", "rawvideo", "-pix_fmt", "yuv420p"],
                5,
                1920,
                1080,
                0,
                "/dev/dri/renderD128",
            )
            args_str = " ".join(args)
            assert "scale_vaapi" in args_str

    def test_ffmpeg_qsv_xe_fallback_can_be_disabled(self):
        with patch.object(_gpu_selector, "is_xe_driver", return_value=True), patch.dict(
            "os.environ", {"FRIGATE_XE_ALLOW_QSV": "1"}, clear=False
        ):
            args = parse_preset_hardware_acceleration_decode(
                "preset-intel-qsv-h264",
                5,
                1920,
                1080,
                0,
                "/dev/dri/renderD128",
            )
            args_str = " ".join(args)
            assert "-qsv_device /dev/dri/renderD128" in args_str
            assert "-hwaccel vaapi" not in args_str


if __name__ == "__main__":
    unittest.main(verbosity=2)

import os
from unittest.mock import MagicMock

import pytest

from tests.conftest import SYNTH_PKG_DIR, load_module_from_path

app_profiles = load_module_from_path(
    "dengjen_neural_voices.domain._app_profiles_under_test",
    os.path.join(SYNTH_PKG_DIR, "domain", "app_profiles.py"),
    package="dengjen_neural_voices.domain",
)
AppProfileManager = app_profiles.AppProfileManager


class FakeProfile(dict):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name


class FakeConfig:
    def __init__(self):
        self._dirtyProfiles = set()
        self.profiles = [FakeProfile(name="base")]
        self.conf_dict = {"speech": {"dengjen_neural_voices": {}}}

    def _markWriteProfileDirty(self):
        if len(self.profiles) > 1:
            self._dirtyProfiles.add(self.profiles[-1].name)

    def __getitem__(self, key):
        return self.conf_dict[key]

    def isSet(self, key):
        return key in self.conf_dict


@pytest.fixture
def mock_nvda_config(monkeypatch):
    import config

    fake = FakeConfig()
    monkeypatch.setattr(config, "conf", fake)
    return fake


class TestAppProfileManager:
    def test_list_profiles_excludes_many_wildcard(self, mock_nvda_config):
        mgr = AppProfileManager()
        section = mgr._profiles_section()
        section["__many__"] = {"voice": "default", "rate": 50}
        section["notepad.exe"] = {"voice": "piper:test", "rate": 60}

        profiles = mgr.list_profiles()
        names = [name for name, _ in profiles]
        assert "__many__" not in names
        assert "notepad.exe" in names

    def test_get_profile_rejects_many_wildcard(self, mock_nvda_config):
        mgr = AppProfileManager()
        section = mgr._profiles_section()
        section["__many__"] = {"voice": "default"}
        assert mgr.get_profile("__many__") == {}
        assert mgr.get_profile("") == {}

    def test_ensure_section_spec_copies_from_many(self, mock_nvda_config):
        mgr = AppProfileManager()
        section = MagicMock()
        section._spec = {
            "__many__": {
                "rate": "integer(min=0, max=100)",
                "volume": "integer(min=0, max=100)",
            }
        }
        mgr._ensure_section_spec(section, "code.exe")
        assert "code.exe" in section._spec
        assert section._spec["code.exe"] == section._spec["__many__"]

    def test_set_profile_none_removes_key_and_marks_dirty(
        self, mock_nvda_config, monkeypatch
    ):
        mgr = AppProfileManager()
        dirty_called = False

        def fake_mark_dirty():
            nonlocal dirty_called
            dirty_called = True

        import config

        monkeypatch.setattr(config.conf, "_markWriteProfileDirty", fake_mark_dirty)

        mgr.set_profile("app.exe", voice="v1", rate=50)
        assert mgr.get_profile("app.exe")["voice"] == "v1"

        mgr.set_profile("app.exe", voice=None)
        assert "voice" not in mgr.get_profile("app.exe")
        assert dirty_called

    def test_delete_profile_removes_from_all_profiles_and_marks_dirty(
        self, mock_nvda_config
    ):
        import config

        p1 = FakeProfile(name="profile1")
        p1["speech"] = {
            "dengjen_neural_voices": {"app_profiles": {"app.exe": {"rate": 50}}}
        }
        p2 = FakeProfile(name="profile2")
        p2["speech"] = {
            "dengjen_neural_voices": {"app_profiles": {"app.exe": {"rate": 60}}}
        }
        config.conf.profiles = [p1, p2]

        mgr = AppProfileManager()
        mgr.set_profile("app.exe", rate=70)

        mgr.delete_profile("app.exe")
        assert mgr.get_profile("app.exe") == {}
        assert "app.exe" not in p1["speech"]["dengjen_neural_voices"]["app_profiles"]
        assert "app.exe" not in p2["speech"]["dengjen_neural_voices"]["app_profiles"]
        assert "profile1" in config.conf._dirtyProfiles
        assert "profile2" in config.conf._dirtyProfiles

    def test_apply_profile_dict_clamps_and_handles_bad_values(self, monkeypatch):
        mgr = AppProfileManager()
        synth = MagicMock()
        synth.voice = "old_voice"

        debug_mock = MagicMock()
        monkeypatch.setattr(app_profiles.log, "debug", debug_mock)

        profile = {
            "voice": "new_voice",
            "rate": "150",
            "volume": "-20",
            "pitch": "invalid_number",
        }
        res = mgr.apply_profile_dict(profile, synth)
        assert res is True
        assert synth.voice == "new_voice"
        assert synth.rate == 100
        assert synth.volume == 0
        debug_mock.assert_called_once()
        assert "invalid pitch" in debug_mock.call_args[0][0]

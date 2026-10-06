"""Config v50: the bundled tirith scanner moved to an opt-in plugin."""

import yaml


class TestRetiredTirithKeys:
    def test_v50_drops_tirith_keys_without_enabling_the_plugin(self, tmp_path, monkeypatch, capsys):
        """The scanner left core: its keys are dropped, nothing re-enables it, and an operator who
        had opted into stricter scanning (fail-closed / a custom path) is told where it went."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 49,
            "security": {"redact_secrets": True, "tirith_enabled": True, "tirith_fail_open": False},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(49, {"env_added": [], "config_added": [], "warnings": []}, quiet=False)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["security"] == {"redact_secrets": True}
        assert "tirith" not in (raw.get("plugins") or {}).get("enabled", [])
        assert "hermes plugins install tirith" in capsys.readouterr().out
        assert not any(key.startswith("tirith") for key in DEFAULT_CONFIG["security"])

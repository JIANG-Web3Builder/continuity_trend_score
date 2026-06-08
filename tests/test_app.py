import importlib


def test_app_import_does_not_require_streamlit_dependency():
    app = importlib.import_module("app")

    assert hasattr(app, "run_dashboard")

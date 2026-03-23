import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
)

import db.database
from unittest.mock import patch, MagicMock

# Mock dependencies to avoid ModuleNotFoundError when importing manage_session
sys.modules["tensorflow"] = MagicMock()
sys.modules["email_validator"] = MagicMock()

# Mock pydantic version check
import pydantic.networks

import builtins
real_open = builtins.open


def test_model_loads_successfully():
    def mock_open(file, *args, **kwargs):
        if "fake_scale.pkl" in str(file):
            return MagicMock()
        return real_open(file, *args, **kwargs)

    with patch("tensorflow.keras.models.load_model") as mock_load, \
         patch("pickle.load", return_value=MagicMock()) as mock_scaler, \
         patch("builtins.open", side_effect=mock_open), \
         patch("pydantic.networks.version", return_value="2.0.0"):

        mock_load.return_value = MagicMock()

        # Mock environment variables
        with patch.dict(os.environ, {"shoulder_model": "fake_model.h5", "scale": "fake_scale.pkl"}):
            import routers.manage_session as session_router
            import importlib
            importlib.reload(session_router)
            
            # Now call the lazy loader
            model, scaler = session_router.get_model_and_scaler()

            assert mock_load.called
            assert mock_scaler.called
            assert model is not None
            assert scaler is not None

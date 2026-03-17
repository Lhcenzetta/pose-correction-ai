import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

import db.database  
from unittest.mock import patch, MagicMock

def test_model_loads_successfully():
    with patch("tensorflow.keras.models.load_model") as mock_load, \
         patch("builtins.open", MagicMock()), \
         patch("pickle.load", return_value=MagicMock()) as mock_scaler:

        mock_load.return_value = MagicMock()  


        import importlib
        import routers.manage_session as session_router
        importlib.reload(session_router)

      
        # It might be called more than once due to module-level loading and reload
        assert mock_load.called
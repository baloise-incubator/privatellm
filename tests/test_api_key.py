import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import call, patch

from privatellm.main import assert_api_key


class ApiKeyTest(unittest.TestCase):
    def test_empty(self):
        with patch.dict("os.environ", {}), patch("privatellm.main.Path", spec_set=Path) as MockPath:
            MockPath.return_value = MockPath
            MockPath.exists.return_value = False
            try:
                assert_api_key()
                self.fail("An assertion should have be thrown")
            except AssertionError as exp:
                assert exp.args == ("Missing OpenAI API key.",)
            assert MockPath.call_args_list == [call("apikey.txt")]

    def test_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fooenv"}):
            assert assert_api_key() == "fooenv"

    def test_file(self):
        with patch.dict("os.environ", {}), patch("privatellm.main.Path", spec_set=Path) as MockPath:
            MockPath.return_value = MockPath
            MockPath.exists.return_value = True
            MockPath.open.return_value = StringIO(" foofile  ")  # expect stripped value
            assert assert_api_key() == "foofile"

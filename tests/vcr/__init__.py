import os
from hashlib import sha256

import pytest
from pytest_mock import MockerFixture

from .vcr_filter import (
    PROXY,
    before_record_request,
    before_record_response,
    freeze_request_body,
)

DEFAULT_RECORD_MODE = "new_episodes"
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
    "filter_headers": [
        "Authorization",
        "AGENT-MODE",
        "AGENT-TYPE",
        "OpenAI-Organization",
        "X-OpenAI-Client-User-Agent",
        "User-Agent",
    ],
    "match_on": ["method", "headers"],
}


@pytest.fixture(scope="session")
def vcr_config(get_base_vcr_config):
    return get_base_vcr_config


@pytest.fixture(scope="session")
def get_base_vcr_config(request):
    record_mode = request.config.getoption("--record-mode", default="new_episodes")
    config = BASE_VCR_CONFIG

    if record_mode is None:
        config["record_mode"] = DEFAULT_RECORD_MODE

    return config


@pytest.fixture()
def vcr_cassette_dir(request):
    test_name = os.path.splitext(request.node.name)[0]
    return os.path.join("tests/Auto-GPT-test-cassettes", test_name)


@pytest.fixture
def patched_api_requestor(mocker: MockerFixture):
    """No-op under BitNet (no OpenAI HTTP client). Kept for challenge test fixtures."""
    yield

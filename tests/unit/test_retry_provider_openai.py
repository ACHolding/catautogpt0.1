import pytest

from autogpt.llm.providers import openai


def error_factory(error_instance, error_count, retry_count, warn_user=True):
    """Creates a callable that fails a few times then succeeds."""

    class RaisesError:
        def __init__(self):
            self.count = 0

        @openai.retry_api(
            max_retries=retry_count, backoff_base=0.001, warn_user=warn_user
        )
        def __call__(self):
            self.count += 1
            if self.count <= error_count:
                raise error_instance
            return self.count

    return RaisesError()


def test_retry_bitnet_no_error(capsys):
    @openai.retry_api()
    def f():
        return 1

    assert f() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


@pytest.mark.parametrize(
    "error_count, retry_count, failure",
    [(2, 10, False), (2, 2, False), (10, 2, True), (3, 2, True), (1, 0, True)],
    ids=["passing", "passing_edge", "failing", "failing_edge", "failing_no_retries"],
)
def test_retry_bitnet_passing(capsys, error_count, retry_count, failure):
    call_count = min(error_count, retry_count) + 1
    raises = error_factory(RuntimeError("boom"), error_count, retry_count)
    if failure:
        with pytest.raises(RuntimeError):
            raises()
    else:
        assert raises() == call_count

    assert raises.count == call_count

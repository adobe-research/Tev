from unittest.mock import patch

import pytest

from open_instruct.environments import pool


class _FakeRemoteMethod:
    def remote(self):
        return object()


class _FakeActor:
    def __init__(self):
        self.setup = _FakeRemoteMethod()


class _FakeRemoteClass:
    def __init__(self):
        self.actors: list[_FakeActor] = []

    def remote(self, **kwargs):
        actor = _FakeActor()
        self.actors.append(actor)
        return actor


class _FakeActorDiedError(Exception):
    pass


def test_pool_initialization_rebuilds_all_actors_after_actor_death():
    remote_class = _FakeRemoteClass()
    actor_died = _FakeActorDiedError("worker exited with SIGSEGV")

    with (
        patch.object(pool.ray.exceptions, "ActorDiedError", _FakeActorDiedError),
        patch.object(pool.ray, "get", side_effect=[actor_died, None]) as ray_get,
        patch.object(pool.ray, "kill") as ray_kill,
        patch.object(pool.time, "sleep") as sleep,
    ):
        actors = pool._create_actors_with_retries(remote_class, 3, {"backend": "docker"}, 3, 2.0)

    assert actors == remote_class.actors[3:]
    assert ray_get.call_count == 2
    assert [call.args[0] for call in ray_kill.call_args_list] == remote_class.actors[:3]
    assert all(call.kwargs == {"no_restart": True} for call in ray_kill.call_args_list)
    sleep.assert_called_once_with(2.0)


def test_pool_initialization_raises_after_retry_budget_is_exhausted():
    remote_class = _FakeRemoteClass()

    with (
        patch.object(pool.ray.exceptions, "ActorDiedError", _FakeActorDiedError),
        patch.object(pool.ray, "get", side_effect=_FakeActorDiedError("worker died")) as ray_get,
        patch.object(pool.ray, "kill") as ray_kill,
        patch.object(pool.time, "sleep") as sleep,
        pytest.raises(_FakeActorDiedError, match="worker died"),
    ):
        pool._create_actors_with_retries(remote_class, 2, {}, 3, 1.0)

    assert ray_get.call_count == 3
    assert ray_kill.call_count == 6
    assert sleep.call_count == 2


def test_pool_initialization_does_not_retry_setup_errors():
    remote_class = _FakeRemoteClass()

    with (
        patch.object(pool.ray.exceptions, "ActorDiedError", _FakeActorDiedError),
        patch.object(pool.ray, "get", side_effect=ValueError("invalid setup")) as ray_get,
        patch.object(pool.ray, "kill") as ray_kill,
        pytest.raises(ValueError, match="invalid setup"),
    ):
        pool._create_actors_with_retries(remote_class, 2, {}, 3, 1.0)

    ray_get.assert_called_once()
    ray_kill.assert_not_called()

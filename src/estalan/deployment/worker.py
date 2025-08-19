import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TypedDict, cast

import structlog
from langgraph.pregel.debug import CheckpointPayload, TaskResultPayload
from starlette.exceptions import HTTPException

import langgraph_api.logging as lg_logging
from langgraph_api.auth.custom import SimpleUser, normalize_user
from estalan.deployment.config import (
    BG_JOB_ISOLATED_LOOPS,
    BG_JOB_MAX_RETRIES,
    BG_JOB_TIMEOUT_SECS,
)
from langgraph_api.errors import UserInterrupt, UserRollback, UserTimeout
from langgraph_api.js.errors import RemoteException

from langgraph_api.schema import Run
from langgraph_api.state import state_snapshot_to_thread_state
from estalan.deployment.stream import AnyStream, astream_state, consume
from langgraph_api.utils import with_user
from langgraph_runtime.database import connect
from langgraph_runtime.ops import Runs, Threads
from langgraph_runtime.retry import RETRIABLE_EXCEPTIONS

logger = structlog.stdlib.get_logger(__name__)

ALL_RETRIABLE_EXCEPTIONS = (asyncio.CancelledError, *RETRIABLE_EXCEPTIONS)


class WorkerResult(TypedDict):
    checkpoint: CheckpointPayload | None
    status: str | None
    exception: Exception | None
    run: Run
    webhook: str | None
    run_started_at: str
    run_ended_at: str | None


@asynccontextmanager
async def set_auth_ctx_for_run(
    run_kwargs: dict, user_id: str | None = None
) -> AsyncGenerator[None, None]:
    # user_id is a fallback.
    try:
        user = run_kwargs["config"]["configurable"]["langgraph_auth_user"]
        permissions = run_kwargs["config"]["configurable"]["langgraph_auth_permissions"]
        user = normalize_user(user)
        # Reapply normalization to the kwargs
        run_kwargs["config"]["configurable"]["langgraph_auth_user"] = user
    except Exception:
        user = SimpleUser(user_id) if user_id is not None else None
        permissions = None
    if user is not None:
        async with with_user(user, permissions):
            yield None
    else:
        yield None


async def worker(
    run: Run,
    attempt: int,
    main_loop: asyncio.AbstractEventLoop,
) -> WorkerResult:
    run_id = run["run_id"]
    checkpoint: CheckpointPayload | None = None
    exception: Exception | None = None
    status: str | None = None
    webhook = run["kwargs"].get("webhook", None)
    request_created_at: int | None = run["kwargs"]["config"]["configurable"].get(
        "__request_start_time_ms__"
    )
    after_seconds = run["kwargs"]["config"]["configurable"].get("__after_seconds__", 0)
    run_started_at_dt = datetime.now(UTC)
    run_started_at = run_started_at_dt.isoformat()
    run_ended_at_dt: datetime | None = None
    run_ended_at: str | None = None

    # Note that "created_at" is inclusive of the `after_seconds`
    run_creation_ms = (
        int(
            ((run["created_at"].timestamp() - after_seconds) * 1_000)
            - request_created_at
        )
        if request_created_at is not None
        else None
    )
    temporary = run["kwargs"].get("temporary", False)
    resumable = run["kwargs"].get("resumable", False)
    run_created_at = run["created_at"].isoformat()
    lg_logging.set_logging_context(
        {
            "run_id": str(run_id),
            "run_attempt": attempt,
            "thread_id": str(run.get("thread_id")),
            "assistant_id": str(run.get("assistant_id")),
            "graph_id": _get_graph_id(run),
            "request_id": _get_request_id(run),
        }
    )
    run_stream_started_at_dt = datetime.now(UTC)
    await logger.ainfo(
        "Starting background run",
        run_started_at=run_started_at,
        run_creation_ms=run_creation_ms,
        run_queue_ms=ms(run_started_at_dt, run["created_at"]),
        run_stream_start_ms=ms(run_stream_started_at_dt, run_started_at_dt),
    )

    def on_checkpoint(checkpoint_arg: CheckpointPayload):
        nonlocal checkpoint
        checkpoint = checkpoint_arg

    def on_task_result(task_result: TaskResultPayload):
        if checkpoint is not None:
            for task in checkpoint["tasks"]:
                if task["id"] == task_result["id"]:
                    task.update(task_result)
                    break

    # Wrap the graph execution to separate user errors from server errors
    async def wrap_user_errors(stream: AnyStream, run_id: str, resumable: bool):
        try:
            await consume(stream, run_id, resumable)
        except Exception as e:
            if not isinstance(e, UserRollback | UserInterrupt):
                logger.error(
                    f"Run encountered an error in graph: {type(e)}({e})",
                    exc_info=e,
                )
            # TimeoutError is a special case where we rely on asyncio.wait_for to timeout runs
            # Convert user TimeoutErrors to a custom class so we can distinguish and later convert back
            if isinstance(e, TimeoutError):
                raise UserTimeout(e) from e
            raise

    async with Runs.enter(run_id, main_loop) as done:
        # attempt the run
        try:
            if attempt > BG_JOB_MAX_RETRIES:
                await logger.aerror(
                    "Run exceeded max attempts",
                    run_id=run["run_id"],
                    run_completed_in_ms=(
                        int((time.time() * 1_000) - request_created_at)
                        if request_created_at is not None
                        else None
                    ),
                )

                error_message = (
                    f"Run {run['run_id']} exceeded max attempts ({BG_JOB_MAX_RETRIES}).\n\n"
                    "This may happen if your code blocks the event loop with synchronous I/O bound calls (network requests, database queries, etc.).\n\n"
                    "If that is the case, your issues may be resolved by converting synchronous operations to async (e.g., use aiohttp instead of requests).\n\n"
                )

                if not BG_JOB_ISOLATED_LOOPS:
                    error_message += (
                        "Also consider setting BG_JOB_ISOLATED_LOOPS=true in your environment. This will isolate I/O-bound operations to avoid"
                        " blocking the main API server.\n\n"
                        "See: https://langchain-ai.github.io/langgraph/cloud/reference/env_var/#bg_job_isolated_loops\n\n"
                    )

                raise RuntimeError(error_message)
            async with set_auth_ctx_for_run(run["kwargs"]):
                if temporary:
                    stream = astream_state(cast(Run, run), attempt, done)
                else:
                    stream = astream_state(
                        cast(Run, run),
                        attempt,
                        done,
                        on_checkpoint=on_checkpoint,
                        on_task_result=on_task_result,
                    )
                await asyncio.wait_for(
                    wrap_user_errors(stream, run_id, resumable),
                    BG_JOB_TIMEOUT_SECS,
                )
        except (Exception, asyncio.CancelledError) as ee:
            exception = ee
        except BaseException as eee:
            await logger.aerror(
                "Bubbling failed background run",
                run_id=str(run_id),
                exception_type=str(type(eee)),
                exception=str(eee),
            )
            raise
        finally:
            run_ended_at_dt = datetime.now(UTC)
            run_ended_at = run_ended_at_dt.isoformat()

        # handle exceptions and set status
        async with connect() as conn:
            log_info = {
                "run_id": str(run_id),
                "run_attempt": attempt,
                "run_created_at": run_created_at,
                "run_started_at": run_started_at,
                "run_ended_at": run_ended_at,
                "run_exec_ms": ms(run_ended_at_dt, run_started_at_dt),
                "run_completed_in_ms": (
                    int((run_ended_at_dt.timestamp() * 1_000) - request_created_at)
                    if request_created_at is not None
                    else None
                ),
            }

            if exception is None:
                status = "success"

                await logger.ainfo(
                    "Background run succeeded",
                    **log_info,
                )
                # If a stateful run succeeded but no checkpoint was returned, likely
                # there was a retriable exception that resumed right at the end
                if checkpoint is None and not temporary:
                    await logger.ainfo(
                        "Fetching missing checkpoint for webhook",
                        run_id=str(run_id),
                        run_attempt=attempt,
                    )
                    try:
                        state_snapshot = await Threads.State.get(
                            conn, run["kwargs"]["config"]
                        )
                        checkpoint = state_snapshot_to_thread_state(state_snapshot)
                    except Exception:
                        await logger.aerror(
                            "Failed to fetch missing checkpoint for webhook. Continuing...",
                            exc_info=True,
                            run_id=str(run_id),
                            run_attempt=attempt,
                        )
                await Threads.set_joint_status(
                    conn, run["thread_id"], run_id, status, checkpoint=checkpoint
                )
            elif isinstance(exception, TimeoutError):
                status = "timeout"
                await logger.awarning(
                    "Background run timed out",
                    **log_info,
                )
                await Threads.set_joint_status(
                    conn, run["thread_id"], run_id, status, checkpoint=checkpoint
                )
            elif isinstance(exception, UserRollback):
                status = "rollback"
                try:
                    await Threads.set_joint_status(
                        conn, run["thread_id"], run_id, status, checkpoint=checkpoint
                    )
                    await logger.ainfo(
                        "Background run rolled back",
                        **log_info,
                    )
                except HTTPException as e:
                    if e.status_code == 404:
                        await logger.ainfo(
                            "Ignoring rollback error for missing run",
                            **log_info,
                        )
                    else:
                        raise

                checkpoint = None  # reset the checkpoint
            elif isinstance(exception, UserInterrupt):
                status = "interrupted"
                await logger.ainfo(
                    "Background run interrupted",
                    **log_info,
                )
                await Threads.set_joint_status(
                    conn, run["thread_id"], run_id, status, checkpoint, exception
                )
            elif isinstance(exception, ALL_RETRIABLE_EXCEPTIONS):
                status = "retry"
                await logger.awarning(
                    f"Background run failed, will retry. Exception: {type(exception)}({exception})",
                    **log_info,
                )
                # Don't update thread status yet.
                await Runs.set_status(conn, run_id, "pending")
            else:
                status = "error"

                # Convert UserTimeout to TimeoutError for customers
                if isinstance(exception, UserTimeout):
                    exception = exception.timeout_error

                await logger.aexception(
                    f"Background run failed. Exception: {type(exception)}({exception})",
                    exc_info=not isinstance(exception, RemoteException),
                    **log_info,
                )
                await Threads.set_joint_status(
                    conn, run["thread_id"], run_id, status, checkpoint, exception
                )

            # delete or set status of thread
            if not isinstance(exception, ALL_RETRIABLE_EXCEPTIONS):
                if temporary:
                    await Threads.delete(conn, run["thread_id"])
                else:
                    try:
                        await Threads.set_status(
                            conn, run["thread_id"], checkpoint, exception
                        )
                    except HTTPException as e:
                        if e.status_code == 404:
                            await logger.ainfo(
                                "Ignoring set_status error for missing thread",
                                exc=str(e),
                            )
                        else:
                            raise

        if isinstance(exception, ALL_RETRIABLE_EXCEPTIONS):
            await logger.awarning("RETRYING", exc_info=exception)
            # re-raise so Runs.enter knows not to mark as done
            # Runs.enter will catch the exception, but what triggers the retry
            # is setting the status to "pending"
            raise exception

    return WorkerResult(
        checkpoint=checkpoint,
        status=status,
        exception=exception,
        run=run,
        webhook=webhook,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
    )


def ms(after: datetime, before: datetime) -> int:
    return int((after - before).total_seconds() * 1000)


def _get_request_id(run: Run) -> str | None:
    try:
        return run["kwargs"]["config"]["configurable"]["langgraph_request_id"]
    except Exception:
        return None


def _get_graph_id(run: Run) -> str | None:
    try:
        return run["kwargs"]["config"]["configurable"]["graph_id"]
    except Exception:
        logger.info(f"Failed to get graph_id from run {run['run_id']}")
        return "Unknown"

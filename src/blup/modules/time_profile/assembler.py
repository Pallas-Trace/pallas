from __future__ import annotations

from collections import defaultdict

from blup.data_model import QuantaQuery, QuantaBundle, as_token_key
from blup.modules.interface import WorkJob
from blup.modules.time_profile.types import (
    TimeProfileUpdate,
    TimeProfileJob,
    TimeProfileRequest,
    TimeProfileResult,
)
from blup.state import TraceMode


def _format_proportion(proportion: float) -> str:
    pct = proportion * 100
    if pct >= 10:
        return f"{pct:.1f}%"
    if pct >= 1:
        return f"{pct:.2f}%"
    return f"{pct:.3f}%"

def _format_duration(exclusive_s: float) -> str:
    if exclusive_s >= 1.0:
        return f"{exclusive_s:.3g} s"
    return f"{exclusive_s * 1000:.3g} ms"

def _empty_source() -> dict:
    return {
        "left":             [],
        "right":            [],
        "top":              [],
        "bottom":           [],
        "color":            [],
        "token_key":        [],
        "token_name":       [],
        "token_type":       [],
        "token_id":         [],
        "thread":           [],
        "proportion":       [],
        "exclusive_s":      [],
        "pct_display":      [],
        "time_display":     [],
    }


class TimeProfileAssembler:

    def __init__(self) -> None:
        pass

    def prepare_request(self, update: TimeProfileUpdate) -> TimeProfileRequest:
        thread_centers = {
            name: len(update.active_thread_names) - 0.5 - i
            for i, name in enumerate(update.active_thread_names)
        }

        jobs: list[TimeProfileJob] = []
        for trace_side, trace_ctx in update.context.trace_context.items():
            for thread in update.active_thread_names:
                thread_id = trace_ctx.thread_name_to_id.get(thread)
                if thread_id is None:
                    continue

                jobs.append(
                    TimeProfileJob(
                        thread_name     = thread,
                        thread_id       = thread_id,
                        trace_side      = trace_side,
                        thread_center   = thread_centers[thread],
                        update          = update,
                    )
                )

        return TimeProfileRequest(
            request_key     = update.request_key,
            jobs            = tuple(jobs),
        )

    def make_job_list(
        self,
        request: TimeProfileRequest
    ) -> list[WorkJob[TimeProfileJob]]:
        return [
            WorkJob(id=i, payload=job)
            for i, job in enumerate(request.jobs)
        ]

    def run_job(self, job: WorkJob[TimeProfileJob]) -> TimeProfileResult:
        payload = job.payload
        update = payload.update
        ctx = update.context
        trace_ctx = ctx.trace_context[payload.trace_side]

        query = QuantaQuery(
            thread_ids      = (payload.thread_id,),
            bin_edges_ns    = ctx.bin_edges_ns,
            fidelity        = ctx.fidelity,
            token_mode      = ctx.token_mode,
            top_k           = -1,
        )
        bundle = trace_ctx.query_quanta(query)

        src = self._build_source(
            bundle,
            thread_name = payload.thread_name,
            thread_center = payload.thread_center,
            trace_side = payload.trace_side,
            token_name_by_key = trace_ctx.token_name_by_key,
            color_map = ctx.color_map,
            trace_mode = update.trace_mode,
            stack_order = ctx.order,
        )

        return TimeProfileResult(
            thread_name = payload.thread_name,
            trace_side = payload.trace_side,
            src = src,
        )

    def _build_source(
        self,
        bundle: QuantaBundle,
        *,
        thread_name: str,
        thread_center: float,
        trace_side: str,
        token_name_by_key: dict[str, str],
        color_map: dict[str, str],
        trace_mode: TraceMode = "dual",
        stack_order: str = "global",
    ) -> dict:
        if len(bundle.start_ns) == 0:
            return _empty_source()

        start_ns_arr = bundle.start_ns
        end_ns_arr = bundle.end_ns
        token_type_arr = bundle.token_type
        token_id_arr = bundle.token_id
        proportion_arr = bundle.proportion
        excl_ns_arr = bundle.excl_ns

        OTHER_TOKEN_TYPE = 255
        OTHER_TOKEN_ID = 0
        OTHER_TOKEN_KEY = "OTHER"
        OTHER_TOKEN_NAME = "OTHER"

        dual_half = 0.45
        dual_padding = 0.02
        single_half = 0.45

        rows: list[tuple[float, float, str, str, int, int, float, float]] = []
        for i in range(len(start_ns_arr)):
            token_type = int(token_type_arr[i])
            token_id = int(token_id_arr[i])

            if token_type == OTHER_TOKEN_TYPE and token_id == OTHER_TOKEN_ID:
                token_key = OTHER_TOKEN_KEY
                token_name = OTHER_TOKEN_NAME
            else:
                token_key = as_token_key(int(token_type), int(token_id))
                token_name = token_name_by_key.get(token_key, token_key)

            rows.append(
                (
                    int(start_ns_arr[i]) / 1e6,
                    int(end_ns_arr[i]) / 1e6,
                    token_key,
                    token_name if token_name else "",
                    token_type,
                    token_id,
                    float(proportion_arr[i]),
                    int(excl_ns_arr[i]) / 1e9,
                )
            )


        if not rows:
            return _empty_source()

        if stack_order == "global":
            totals: dict[str, float] = defaultdict(float)
            for _, _, token_key, _, _, _, _, exclusive_s in rows:
                totals[token_key] += exclusive_s

            rank = {
                token_key: i
                for i, (token_key, _) in enumerate(
                    sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
                )
            }
            sort_key_fn = lambda row: rank.get(row[2], 10**9)
        elif stack_order == "local":
            sort_key_fn = lambda row: (-row[6], row[2])
        else:
            raise ValueError(f"invalid stack_order: {stack_order!r}")

        grouped: (
            dict[tuple[float, float],
            list[tuple[str, str, int, int, float, float]]]
        ) = defaultdict(list)
        for (
            left, right, token_key, token_name,
            token_type, token_id, proportion, exclusive_s
        ) in rows:
            grouped[(left, right)].append(
                (token_key, token_name, token_type,
                 token_id, proportion, exclusive_s)
            )

        out = _empty_source()
        centered_mode = (trace_mode != "dual")
        dual_half = 0.45
        dual_padding = 0.02
        single_half = 0.45

        for (left, right), grp in grouped.items():
            grp = sorted(grp, key=sort_key_fn)
            cumsum = 0.0

            for (
                token_key, token_name, token_type,
                token_id, proportion, exclusive_s
            ) in grp:
                if centered_mode:
                    lane_bottom = thread_center - single_half
                    lane_top = thread_center + single_half
                    lane_height = lane_top - lane_bottom
                    bottom = lane_bottom + cumsum * lane_height
                    top = lane_bottom + (cumsum + proportion) * lane_height
                else:
                    if trace_side == "lower":
                        top = (thread_center - dual_padding
                            - cumsum * dual_half)
                        bottom = (thread_center - dual_padding
                            - (cumsum + proportion) * dual_half)
                    else:
                        bottom = (thread_center + dual_padding
                            + cumsum * dual_half)
                        top = (thread_center + dual_padding
                            + (cumsum + proportion) * dual_half)

                out["left"].append(left)
                out["right"].append(right)
                out["top"].append(top)
                out["bottom"].append(bottom)
                out["color"].append(color_map.get(token_key, "#999999"))
                out["token_key"].append(token_key)
                out["token_name"].append(token_name)
                out["token_type"].append(token_type)
                out["token_id"].append(token_id)
                out["thread"].append(thread_name)
                out["proportion"].append(proportion)
                out["exclusive_s"].append(exclusive_s)
                out["pct_display"].append(_format_proportion(proportion))
                out["time_display"].append(_format_duration(exclusive_s))

                cumsum += proportion

        return out



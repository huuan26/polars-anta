"""Binder helpers and declarative frame-spec builders for Polars namespaces."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import polars as pl

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

_EMPTY = inspect.Parameter.empty
_PK = inspect.Parameter.POSITIONAL_OR_KEYWORD
_SELF_PARAM = inspect.Parameter("self", _PK)
_FRAME_CTRL_KEYS = frozenset({"alias", "unnest", "prefix", "group_by"})
_EXPR_ALPHA_DEFAULTS: dict[str, str] = {
    "open_": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
}
_FRAME_ALPHA_DEFAULTS: dict[str, str] = {**_EXPR_ALPHA_DEFAULTS, "close": "close"}


def _bind_expr(
    fn: Callable[..., pl.Expr],
    *,
    self_param: str | None = "close",
    renames: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
    params: Sequence[tuple[str, Any]] | None = None,
) -> Callable[..., pl.Expr]:
    """Build a TechnicalAnalysisExpr method forwarding ``self._expr`` to ``fn``."""
    rename_map = dict(renames) if renames else {}
    inv_renames = {v: k for k, v in rename_map.items()}
    if params is not None:
        mparams = [inspect.Parameter(n, _PK, default=d) for n, d in params]
    else:
        mparams = []
        for fp in inspect.signature(fn).parameters.values():
            if fp.name == self_param:
                continue
            m_name = rename_map.get(fp.name, fp.name)
            dflt = (
                defaults[m_name]
                if (fp.default is _EMPTY and defaults and m_name in defaults)
                else fp.default
            )
            mparams.append(inspect.Parameter(m_name, _PK, default=dflt))
    msig = inspect.Signature(parameters=[_SELF_PARAM, *mparams])

    def method(self: Any, *args: Any, **kwargs: Any) -> pl.Expr:
        bound = msig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call_kwargs: dict[str, Any] = {}
        if self_param is not None:
            call_kwargs[self_param] = self._expr
        for m_name, val in list(bound.arguments.items())[1:]:
            call_kwargs[inv_renames.get(m_name, m_name)] = val
        return fn(**call_kwargs)

    method.__name__ = fn.__name__
    method.__qualname__ = f"TechnicalAnalysisExpr.{fn.__name__}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return method


def _bind_frame_passthrough(
    fn: Callable[..., Any],
    name: str | None = None,
) -> tuple[str, Callable[..., Any]]:
    """Build a TechnicalAnalysisFrame method forwarding ``self._df`` to ``fn``."""
    meth_name = name or fn.__name__
    fsig = inspect.signature(fn)
    msig = fsig.replace(parameters=[_SELF_PARAM, *list(fsig.parameters.values())[1:]])

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(self._df, *args, **kwargs)

    method.__name__ = meth_name
    method.__qualname__ = f"TechnicalAnalysisFrame.{meth_name}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return meth_name, method


@dataclass(frozen=True)
class _FrameExprSpec:
    fn: Callable[..., pl.Expr]
    params: tuple[tuple[str, Any], ...]
    renames: Mapping[str, str] | None = None
    alias_fmt: str | Callable[[Mapping[str, Any]], str] | None = None


def _col_period_spec(
    fn: Callable[..., pl.Expr],
    period: int,
    alias_base: str | None = None,
    col_target: str = "close",
) -> _FrameExprSpec:
    base = alias_base or fn.__name__
    return _FrameExprSpec(
        fn=fn,
        params=(("period", period), ("col", "close"), ("alias", None), ("group_by", None)),
        renames={"col": col_target},
        alias_fmt=f"{base}_{{period}}",
    )


def _struct_spec(
    fn: Callable[..., pl.Expr],
    /,
    *,
    _alias: str | None = None,
    _prefix: str | None = None,
    _renames: Mapping[str, str] | None = None,
    **params: Any,
) -> _FrameExprSpec:
    return _FrameExprSpec(
        fn=fn,
        params=(
            *params.items(),
            ("alias", _alias or fn.__name__),
            ("unnest", False),
            ("prefix", _prefix),
            ("group_by", None),
        ),
        renames=_renames,
    )


def _scalar_spec(
    fn: Callable[..., pl.Expr],
    /,
    *,
    _alias_fmt: str | Callable[[Mapping[str, Any]], str] | None = None,
    _alias: str | None = None,
    _renames: Mapping[str, str] | None = None,
    _has_gb: bool = True,
    **params: Any,
) -> _FrameExprSpec:
    tail: tuple[tuple[str, Any], ...] = (
        (("alias", _alias), ("group_by", None)) if _has_gb else (("alias", _alias),)
    )
    return _FrameExprSpec(
        fn=fn,
        params=(*params.items(), *tail),
        renames=_renames,
        alias_fmt=_alias_fmt,
    )


def _bind_frame_expr(spec: _FrameExprSpec) -> tuple[str, Callable[..., Any]]:
    """Build a TechnicalAnalysisFrame indicator method backed by ``_apply_expr``."""
    fn = spec.fn
    meth_name = fn.__name__
    rename_map = dict(spec.renames) if spec.renames else {}
    msig = inspect.Signature(
        parameters=[_SELF_PARAM, *(inspect.Parameter(n, _PK, default=d) for n, d in spec.params)]
    )

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = msig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        raw_args = dict(list(bound.arguments.items())[1:])
        call_kwargs = {
            rename_map.get(k, k): v for k, v in raw_args.items() if k not in _FRAME_CTRL_KEYS
        }
        alias_val = raw_args["alias"]
        if spec.alias_fmt is not None:
            fallback = (
                spec.alias_fmt(raw_args)
                if callable(spec.alias_fmt)
                else spec.alias_fmt.format_map(raw_args)
            )
            alias_name = alias_val or fallback
        else:
            alias_name = alias_val
        return self._apply_expr(
            fn(**call_kwargs),
            alias=alias_name,
            unnest=raw_args.get("unnest", False),
            prefix=raw_args.get("prefix"),
            group_by=raw_args.get("group_by"),
        )

    method.__name__ = meth_name
    method.__qualname__ = f"TechnicalAnalysisFrame.{meth_name}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return meth_name, method

"""Quản lý hard budget USD có checkpoint cho benchmark dùng provider trả phí."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from adaptive_rag_router.domain.enums import RouterKind
from adaptive_rag_router.domain.errors import ConfigurationError, CreditBudgetExceededError
from adaptive_rag_router.evaluation.models import BenchmarkRecord, BenchmarkSample


class BudgetHook(Protocol):
    """Contract tối thiểu của một budget hook cho benchmark runner."""

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Kiểm tra một query-router invocation có được bắt đầu hay không."""

    async def commit(self, record: BenchmarkRecord) -> None:
        """Ghi nhận usage của invocation vừa hoàn tất."""


class UsdLedger(Protocol):
    """Contract ledger đơn giá trị dùng cho một nhóm provider calls."""

    async def reserve(self, unit_id: str, *, reserve_usd: float) -> bool:
        """Kiểm tra còn đủ vùng đệm cho một đơn vị công việc."""

    async def commit(self, unit_id: str, cost_usd: float) -> None:
        """Commit cost idempotent của một đơn vị công việc."""


class FileUsdBudgetLedger:
    """Ledger USD idempotent, ghi atomic sau mỗi đơn vị benchmark.

    Args:
        path: File JSON checkpoint của ledger.
        limit_usd: Tổng chi phí tối đa được phép ghi nhận.

    Raises:
        ConfigurationError: Khi checkpoint hỏng hoặc dùng limit khác run cũ.
    """

    def __init__(self, path: Path, *, limit_usd: float) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd phải dương")
        self._path = path
        self._limit_usd = limit_usd
        self._entries = self._load()
        self._lock = asyncio.Lock()

    @property
    def limit_usd(self) -> float:
        """Trả hard limit USD của run."""

        return self._limit_usd

    @property
    def consumed_usd(self) -> float:
        """Trả tổng cost đã commit, không làm tròn."""

        return sum(self._entries.values())

    @property
    def remaining_usd(self) -> float:
        """Trả phần budget còn lại, chặn dưới tại 0."""

        return max(0.0, self._limit_usd - self.consumed_usd)

    async def reserve(self, unit_id: str, *, reserve_usd: float) -> bool:
        """Kiểm tra còn đủ vùng đệm trước một đơn vị provider calls."""

        if reserve_usd <= 0:
            raise ValueError("reserve_usd phải dương")
        async with self._lock:
            return unit_id in self._entries or self.consumed_usd + reserve_usd <= self._limit_usd

    async def commit(self, unit_id: str, cost_usd: float) -> None:
        """Commit cost theo unit ID; gọi lặp cùng giá trị không cộng hai lần.

        Raises:
            ConfigurationError: Khi cùng unit ID có cost khác checkpoint.
            CreditBudgetExceededError: Khi cost thực tế làm vượt hard limit.
        """

        if cost_usd < 0:
            raise ValueError("cost_usd không được âm")
        async with self._lock:
            existing = self._entries.get(unit_id)
            if existing is not None:
                if abs(existing - cost_usd) > 1e-9:
                    # Failed record được checkpoint với cost 0 và sẽ được runner
                    # retry. Khi retry thành công, thay placeholder 0 thay vì
                    # cộng thêm hoặc từ chối một kết quả hợp lệ.
                    if existing == 0 and cost_usd > 0:
                        self._entries[unit_id] = cost_usd
                        await asyncio.to_thread(self._write)
                        if self.consumed_usd > self._limit_usd + 1e-9:
                            raise CreditBudgetExceededError(
                                f"Cost thực tế ${self.consumed_usd:.6f} vượt hard limit "
                                f"${self._limit_usd:.2f}"
                            )
                        return
                    raise ConfigurationError(f"Cost ledger lệch cho unit {unit_id!r}")
                return
            self._entries[unit_id] = cost_usd
            await asyncio.to_thread(self._write)
            if self.consumed_usd > self._limit_usd + 1e-9:
                raise CreditBudgetExceededError(
                    f"Cost thực tế ${self.consumed_usd:.6f} vượt hard limit ${self._limit_usd:.2f}"
                )

    def _load(self) -> dict[str, float]:
        if not self._path.is_file():
            return {}
        try:
            payload: Any = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload không phải object")
            stored_limit = float(payload["limit_usd"])
            if abs(stored_limit - self._limit_usd) > 1e-9:
                raise ValueError("limit_usd khác checkpoint")
            raw_entries = payload["entries"]
            if not isinstance(raw_entries, dict):
                raise ValueError("entries không phải object")
            return {str(key): float(value) for key, value in raw_entries.items()}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Cost ledger hỏng: {self._path}") from error

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "limit_usd": self._limit_usd,
                    "consumed_usd": self.consumed_usd,
                    "remaining_usd": self.remaining_usd,
                    "entries": self._entries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


class FileProviderBudgetLedger:
    """Ledger project-wide áp hard cap USD riêng cho từng provider.

    Args:
        path: File JSON checkpoint dùng chung giữa mọi benchmark run.
        limits_usd: Hard cap theo provider, ví dụ ``deepseek`` và
            ``typesafe-jev``.

    Raises:
        ValueError: Khi provider hoặc limit không hợp lệ.
        ConfigurationError: Khi checkpoint hỏng hoặc limits bị thay đổi.
    """

    def __init__(self, path: Path, *, limits_usd: Mapping[str, float]) -> None:
        normalized_limits = {
            str(provider).strip(): float(limit) for provider, limit in limits_usd.items()
        }
        if not normalized_limits or any(
            not provider or limit <= 0 for provider, limit in normalized_limits.items()
        ):
            raise ValueError("limits_usd cần provider không rỗng và limit dương")
        self._path = path
        self._limits_usd = normalized_limits
        self._entries = self._load()
        self._lock = asyncio.Lock()

    @property
    def limits_usd(self) -> dict[str, float]:
        """Trả bản sao hard limits theo provider."""

        return dict(self._limits_usd)

    @property
    def path(self) -> Path:
        """Trả đường dẫn checkpoint project-wide."""

        return self._path

    @property
    def consumed_usd(self) -> dict[str, float]:
        """Tổng cost đã commit theo provider."""

        return {
            provider: sum(entry.get(provider, 0.0) for entry in self._entries.values())
            for provider in self._limits_usd
        }

    @property
    def remaining_usd(self) -> dict[str, float]:
        """Budget còn lại theo provider, chặn dưới tại 0."""

        consumed = self.consumed_usd
        return {
            provider: max(0.0, limit - consumed[provider])
            for provider, limit in self._limits_usd.items()
        }

    async def reserve(self, unit_id: str, *, reserves_usd: Mapping[str, float]) -> bool:
        """Chỉ cho phép unit khi mọi provider còn đủ reserve tương ứng.

        Args:
            unit_id: Khóa idempotent duy nhất trên toàn project.
            reserves_usd: Vùng đệm cần giữ theo provider.

        Returns:
            ``True`` khi unit đã tồn tại hoặc mọi reserve nằm trong hard cap.

        Raises:
            ValueError: Khi provider lạ hoặc reserve không dương.
        """

        normalized = self._normalize_costs(reserves_usd, require_positive=True)
        async with self._lock:
            if unit_id in self._entries:
                return True
            consumed = self.consumed_usd
            return all(
                consumed[provider] + reserve <= self._limits_usd[provider]
                for provider, reserve in normalized.items()
            )

    async def commit(self, unit_id: str, costs_usd: Mapping[str, float]) -> None:
        """Commit provider costs idempotent và báo lỗi khi actual vượt cap.

        Args:
            unit_id: Khóa idempotent duy nhất trên toàn project.
            costs_usd: Actual cost theo provider.

        Raises:
            ConfigurationError: Khi cùng unit có cost khác checkpoint.
            CreditBudgetExceededError: Khi actual cost làm vượt một hard cap.
        """

        normalized = self._normalize_costs(costs_usd, require_positive=False)
        complete = {provider: normalized.get(provider, 0.0) for provider in self._limits_usd}
        async with self._lock:
            existing = self._entries.get(unit_id)
            if existing is not None:
                if self._cost_maps_equal(existing, complete):
                    return
                if all(value == 0 for value in existing.values()) and any(
                    value > 0 for value in complete.values()
                ):
                    self._entries[unit_id] = complete
                    await asyncio.to_thread(self._write)
                    self._raise_if_exceeded()
                    return
                raise ConfigurationError(f"Provider cost ledger lệch cho unit {unit_id!r}")
            self._entries[unit_id] = complete
            await asyncio.to_thread(self._write)
            self._raise_if_exceeded()

    def _normalize_costs(
        self,
        values: Mapping[str, float],
        *,
        require_positive: bool,
    ) -> dict[str, float]:
        normalized = {str(provider).strip(): float(value) for provider, value in values.items()}
        unknown = set(normalized) - set(self._limits_usd)
        if unknown:
            raise ValueError(f"provider không có hard cap: {sorted(unknown)}")
        if any(value < 0 for value in normalized.values()):
            raise ValueError("provider cost/reserve không được âm")
        if require_positive and (
            not normalized or any(value <= 0 for value in normalized.values())
        ):
            raise ValueError("provider reserve phải dương")
        return normalized

    @staticmethod
    def _cost_maps_equal(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        return set(left) == set(right) and all(
            abs(float(left[key]) - float(right[key])) <= 1e-9 for key in left
        )

    def _raise_if_exceeded(self) -> None:
        consumed = self.consumed_usd
        exceeded = {
            provider: value
            for provider, value in consumed.items()
            if value > self._limits_usd[provider] + 1e-9
        }
        if exceeded:
            details = ", ".join(
                f"{provider} ${value:.6f}/${self._limits_usd[provider]:.2f}"
                for provider, value in exceeded.items()
            )
            raise CreditBudgetExceededError(
                f"Cost thực tế vượt project provider hard limit: {details}"
            )

    def _load(self) -> dict[str, dict[str, float]]:
        if not self._path.is_file():
            return {}
        try:
            payload: Any = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload không phải object")
            raw_limits = payload["limits_usd"]
            raw_entries = payload["entries"]
            if not isinstance(raw_limits, dict) or not isinstance(raw_entries, dict):
                raise ValueError("limits_usd/entries không phải object")
            stored_limits = {str(key): float(value) for key, value in raw_limits.items()}
            if not self._cost_maps_equal(stored_limits, self._limits_usd):
                raise ValueError("limits_usd khác checkpoint")
            entries: dict[str, dict[str, float]] = {}
            for unit_id, raw_costs in raw_entries.items():
                if not isinstance(raw_costs, dict):
                    raise ValueError("provider cost entry không phải object")
                costs = {str(key): float(value) for key, value in raw_costs.items()}
                if set(costs) != set(self._limits_usd) or any(
                    value < 0 for value in costs.values()
                ):
                    raise ValueError("provider cost entry sai schema")
                entries[str(unit_id)] = costs
            return entries
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"Provider cost ledger hỏng: {self._path}") from error

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "limits_usd": self._limits_usd,
                    "consumed_usd": self.consumed_usd,
                    "remaining_usd": self.remaining_usd,
                    "entries": self._entries,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


class ProviderScopedUsdLedger:
    """Chiếu ledger nhiều provider thành một ledger đơn provider có namespace."""

    def __init__(
        self,
        ledger: FileProviderBudgetLedger,
        *,
        provider: str,
        namespace: str,
    ) -> None:
        """Khởi tạo adapter cho counterfactual hoặc workflow đơn provider."""

        if provider not in ledger.limits_usd:
            raise ValueError(f"provider {provider!r} không có trong project ledger")
        self._ledger = ledger
        self._provider = provider
        self._namespace = namespace.strip(":")

    async def reserve(self, unit_id: str, *, reserve_usd: float) -> bool:
        """Reserve cost cho provider đã chọn."""

        return await self._ledger.reserve(
            self._unit_id(unit_id),
            reserves_usd={self._provider: reserve_usd},
        )

    async def commit(self, unit_id: str, cost_usd: float) -> None:
        """Commit cost cho provider đã chọn."""

        await self._ledger.commit(
            self._unit_id(unit_id),
            {self._provider: cost_usd},
        )

    def _unit_id(self, unit_id: str) -> str:
        return f"{self._namespace}:{unit_id}" if self._namespace else unit_id


class CombinedUsdLedger:
    """Áp reserve/commit đồng thời lên nhiều USD ledger."""

    def __init__(self, *ledgers: UsdLedger) -> None:
        """Khởi tạo từ ít nhất một ledger."""

        if not ledgers:
            raise ValueError("CombinedUsdLedger cần ít nhất một ledger")
        self._ledgers = ledgers

    async def reserve(self, unit_id: str, *, reserve_usd: float) -> bool:
        """Chỉ chấp nhận khi mọi ledger còn đủ reserve."""

        for ledger in self._ledgers:
            if not await ledger.reserve(unit_id, reserve_usd=reserve_usd):
                return False
        return True

    async def commit(self, unit_id: str, cost_usd: float) -> None:
        """Commit idempotent vào mọi ledger."""

        for ledger in self._ledgers:
            await ledger.commit(unit_id, cost_usd)


class UsdBudgetHook:
    """Nối USD ledger vào adaptive benchmark runner.

    Args:
        ledger: Ledger dùng chung giữa mọi phase.
        run_id: Run ID của phase adaptive.
        reserve_usd: Vùng đệm trước mỗi query-router invocation.
    """

    def __init__(
        self,
        ledger: UsdLedger,
        *,
        run_id: str,
        reserve_usd: float,
    ) -> None:
        self._ledger = ledger
        self._run_id = run_id
        self._reserve_usd = reserve_usd

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Trả False khi phần budget còn lại thấp hơn reserve."""

        return await self._ledger.reserve(
            self._unit_id(sample.query_id, router),
            reserve_usd=self._reserve_usd,
        )

    async def commit(self, record: BenchmarkRecord) -> None:
        """Commit cost của record theo khóa idempotent."""

        await self._ledger.commit(
            self._unit_id(record.query_id, record.router),
            record.cost_usd,
        )

    def _unit_id(self, query_id: str, router: RouterKind) -> str:
        return f"adaptive:{self._run_id}:{query_id}:{router.value}"


class ProviderUsdBudgetHook:
    """Áp hard cap project-wide riêng cho DeepSeek và TypeSafe Jev."""

    def __init__(
        self,
        ledger: FileProviderBudgetLedger,
        *,
        run_id: str,
        deepseek_reserve_usd: float,
        jev_reserve_usd: float,
    ) -> None:
        """Khởi tạo provider hook cho adaptive benchmark.

        Args:
            ledger: Project-wide provider ledger.
            run_id: Run ID dùng tạo khóa idempotent.
            deepseek_reserve_usd: Reserve cho generation/router DeepSeek.
            jev_reserve_usd: Reserve bổ sung khi engine là Jev.
        """

        if deepseek_reserve_usd <= 0 or jev_reserve_usd <= 0:
            raise ValueError("provider reserves phải dương")
        self._ledger = ledger
        self._run_id = run_id
        self._deepseek_reserve_usd = deepseek_reserve_usd
        self._jev_reserve_usd = jev_reserve_usd

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Reserve DeepSeek cho mọi workflow và Jev cho riêng Jev router."""

        reserves = {"deepseek": self._deepseek_reserve_usd}
        if router is RouterKind.JEV:
            reserves["typesafe-jev"] = self._jev_reserve_usd
        return await self._ledger.reserve(
            self._unit_id(sample.query_id, router),
            reserves_usd=reserves,
        )

    async def commit(self, record: BenchmarkRecord) -> None:
        """Commit provider breakdown; record cũ thiếu breakdown được tính bảo thủ."""

        raw_costs = record.metadata.get("provider_costs_usd")
        if isinstance(raw_costs, Mapping):
            costs = {
                str(provider): float(value)
                for provider, value in raw_costs.items()
                if str(provider) in self._ledger.limits_usd
            }
        else:
            # Artifact lịch sử chỉ có total cost. Toàn bộ total được charge cho
            # DeepSeek và đồng thời cho Jev nếu đây là Jev workflow; cách này có
            # thể double-count nhưng không thể làm lỏng hard cap của người dùng.
            costs = {"deepseek": record.cost_usd}
            if record.router is RouterKind.JEV:
                costs["typesafe-jev"] = record.cost_usd
        await self._ledger.commit(self._unit_id(record.query_id, record.router), costs)

    def _unit_id(self, query_id: str, router: RouterKind) -> str:
        return f"adaptive:{self._run_id}:{query_id}:{router.value}"


class CombinedBudgetHook:
    """Áp đồng thời nhiều guardrail như Tavily credits và USD cost."""

    def __init__(self, *hooks: BudgetHook) -> None:
        """Khởi tạo từ ít nhất một hook theo thứ tự kiểm tra ổn định."""

        if not hooks:
            raise ValueError("CombinedBudgetHook cần ít nhất một hook")
        self._hooks = hooks

    async def reserve(self, sample: BenchmarkSample, router: RouterKind) -> bool:
        """Chỉ cho phép invocation khi mọi hook đều chấp nhận."""

        for hook in self._hooks:
            if not await hook.reserve(sample, router):
                return False
        return True

    async def commit(self, record: BenchmarkRecord) -> None:
        """Commit record vào mọi ledger cấu thành."""

        for hook in self._hooks:
            await hook.commit(record)

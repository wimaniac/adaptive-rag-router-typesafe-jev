"""Định nghĩa typed switches cho các ablation của adaptive pipeline."""

from pydantic import BaseModel, ConfigDict


class PipelinePolicy(BaseModel):
    """Bật/tắt từng cơ chế adaptive để chạy ablation có kiểm soát.

    Args:
        context_gate_enabled: Có gọi router đánh giá context hay chấp nhận toàn
            bộ evidence ban đầu.
        repair_enabled: Có cho phép quyết định repair và query rewrite hay không.
        strong_fallback_enabled: Có thực thi strong regeneration khi fallback
            gate yêu cầu hay giữ economy draft.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_gate_enabled: bool = True
    repair_enabled: bool = True
    strong_fallback_enabled: bool = True

    @property
    def ablation_id(self) -> str:
        """Trả định danh ổn định của baseline hoặc ablation đang bật."""

        disabled: list[str] = []
        if not self.context_gate_enabled:
            disabled.append("context-gate")
        if not self.repair_enabled:
            disabled.append("repair")
        if not self.strong_fallback_enabled:
            disabled.append("strong-fallback")
        return "baseline" if not disabled else "no-" + "-no-".join(disabled)

    @property
    def is_baseline(self) -> bool:
        """Cho biết mọi cơ chế adaptive đều đang bật."""

        return self.ablation_id == "baseline"

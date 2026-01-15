from transformers import PretrainedConfig


class PromptComplexityConfig(PretrainedConfig):
    model_type = "prompt-complexity"

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        dropout: float = 0.1,
        hidden: int | None = None,
        layernorm_after_pool: bool = True,
        use_projection: bool = False,
        proj_hidden_ratio: float = 1.0,
        output_sigmoid: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.dropout = dropout
        self.hidden = hidden
        self.layernorm_after_pool = layernorm_after_pool
        self.use_projection = use_projection
        self.proj_hidden_ratio = proj_hidden_ratio
        self.output_sigmoid = output_sigmoid

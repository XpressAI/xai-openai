from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component
import openai
import os

@xai_component
class OpenAIAuthorize(Component):
    organization: InArg[str]
    api_key: InArg[str]
    from_env: InArg[bool]

    def __init__(self):
        self.done = False
        self.organization = InArg.empty()
        self.key = InArg.empty()
        self.from_env = InArg.empty()

    def execute(self, ctx) -> None:
        openai.organization = self.organization.value
        if self.from_env.value:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = self.key.value

@xai_component
class OpenAIGetModels(Component):
    models: OutArg[list]
    def __init__(self):
        self.done = False
        self.models = OutArg.empty()

    def execute(self, ctx) -> None:
        self.models.value = openai.Model.list()


@xai_component
class OpenAIGetModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

    def __init__(self):
        self.done = False
        self.model_name = InCompArg.empty()
        self.model = OutArg.empty()

    def execute(self, ctx) -> None:
        self.model.value = openai.Model.retrieve(self.model_name.value)



@xai_component
class OpenAIGenerate(Component):
    model_name: InCompArg[str]
    prompt: InCompArg[str]
    max_tokens: InArg[int]
    temperature: InArg[float]
    count: InArg[int]
    completion: OutArg[str]

    def __init__(self):
        self.done = False
        self.model_name = InCompArg.empty()
        self.prompt = InCompArg.empty()
        self.max_tokens = InArg.empty()
        self.temperature = InArg.empty()
        self.count = InArg.empty()
        self.completion = OutArg.empty()

    def execute(self, ctx) -> None:
        result = openai.Completion.create(
            model=self.model_name.value,
            prompt=self.prompt.value,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 16,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            n=self.count.value if self.count.value is not None else 1
        )

        if self.count.value is None or self.count.value == 1:
            self.completion.value = result['choices'][0]['text']
        else:
            self.completion.value = [r['text'] for r in result['choices']]


@xai_component
class OpenAIEdit(Component):
    model_name: InCompArg[str]
    prompt: InCompArg[str]
    instruction: InCompArg[str]
    count: InArg[int]
    temperature: InArg[float]
    edited: OutArg[any]

    def __init__(self):
        self.done = False
        self.model_name = InCompArg.empty()
        self.prompt = InCompArg.empty()
        self.instruction = InCompArg.empty()
        self.count = InArg.empty()
        self.temperature = InArg.empty()
        self.edited = OutArg.empty()

    def execute(self, ctx) -> None:
        result = openai.Edit.create(
            model=self.model_name.value,
            input=self.prompt.value,
            instruction=self.instruction.value,
            n=self.count.value if self.count.value is not None else 1,
            temperature=self.temperature.value if self.temperature.value is not None else 1
        )

        if self.count.value is None or self.count.value == 1:
            self.edited.value = result['choices'][0]['text']
        else:
            self.edited.value = [r['text'] for r in result['choices']]


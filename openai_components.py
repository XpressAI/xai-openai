from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component
import openai
import os
import requests
import shutil


@xai_component
class OpenAIAuthorize(Component):
    organization: InArg[str]
    api_key: InArg[str]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        openai.organization = self.organization.value
        if self.from_env.value:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = self.api_key.value

@xai_component
class OpenAIGetModels(Component):
    models: OutArg[list]

    def execute(self, ctx) -> None:
        self.models.value = openai.Model.list()


@xai_component
class OpenAIGetModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

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

@xai_component
class OpenAIImageCreate(Component):
    prompt: InCompArg[str]
    image_count: InArg[int]
    size: InArg[str]
    image_urls: OutArg[list]

    def execute(self, ctx) -> None:
        result = openai.Image.create(
            prompt=self.prompt.value,
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d['url'] for d in result['data']]



@xai_component
class DownloadImages(Component):
    image_urls: InCompArg[list]
    file_path: InCompArg[list]
    
    def execute(self, ctx) -> None:
        i = 0
        for image_url in self.image_urls.value:
            response = requests.get(image_url, stream=True)
            with open(self.file_path.value[i], 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            i += 1


@xai_component
class OpenAIImageCreateVariation(Component):
    image_path: InCompArg[str]
    image_count: InArg[int]
    size: InArg[str]
    image_urls: OutArg[list]

    def execute(self, ctx) -> None:
        result = openai.Image.create(
            prompt=self.prompt.value,
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d['url'] for d in result['data']]
        
@xai_component
class TakeNthElement(Component):
    values: InCompArg[list]
    index: InCompArg[int]
    out: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.out.value = self.values.value[self.index.value]

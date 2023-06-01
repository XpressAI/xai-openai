from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import openai
import os
import requests
import shutil


@xai_component
class OpenAIAuthorize(Component):
    """Sets the organization and API key for the OpenAI client.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/authentication)

    ##### inPorts:
    - organization: Organization name id for OpenAI API.
    - api_key: API key for the OpenAI API.
    - from_env: Boolean value indicating whether the API key is to be fetched from environment variables. 
    """
    organization: InArg[secret]
    api_key: InArg[secret]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        openai.organization = self.organization.value
        if self.from_env.value:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = self.api_key.value
        
@xai_component
class OpenAIGetModels(Component):
    """Retrieves a list of all models available from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/models/list)

    ##### inPorts:
    - None: This component does not require any input. 

    ##### outPorts:
    - models: List of available models from OpenAI.
    """

    models: OutArg[list]

    def execute(self, ctx) -> None:
        self.models.value = openai.Model.list()


@xai_component
class OpenAIGetModel(Component):
    """Retrieves a specific model from OpenAI by model name.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/models/retrieve)

    ##### inPorts:
    - model_name: Name of the model to be retrieved.

    ##### outPorts:
    - model: The model retrieved from OpenAI.
    """

    model_name: InCompArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        self.model.value = openai.Model.retrieve(self.model_name.value)



@xai_component
class OpenAIGenerate(Component):
    """Generates text using a specified model from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/completions/create)

    ##### inPorts:
    - model_name: Name of the model to be used for text generation.
    - prompt: The initial text to generate from.
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.
    - count: Number of completions to generate.

    ##### outPorts:
    - completion: The generated text.
    """

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
class OpenAIChat(Component):
    """Interacts with a specified model from OpenAI in a conversation.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/completions/create)

    ##### inPorts:
    - model_name: Name of the model to be used for conversation.
    - system_prompt: Initial system message to start the conversation.
    - user_prompt: Initial user message to continue the conversation.
    - conversation: A list of conversation messages. Each message is a dictionary with a "role" and "content".
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.
    - count: Number of responses to generate.

    ##### outPorts:
    - completion: The generated text of the model's response.
    - out_conversation: The complete conversation including the model's response.
    """
    model_name: InCompArg[str]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    conversation: InArg[list]
    max_tokens: InArg[int]
    temperature: InArg[float]
    count: InArg[int]
    completion: OutArg[str]
    out_conversation: OutArg[list]
        
    def __init__(self):
        super().__init__()
        
    def execute(self, ctx) -> None:
        messages = None
        if self.conversation.value is not None:
            messages = self.conversation.value
        elif self.system_prompt.value is not None:
            messages = [
                {"role": "system", "content": self.system_prompt.value},
                {"role": "user", "content": self.user_prompt.value}
            ]
        else:
            messages.append({"role": "user", "content": self.user_prompt.value })
        
        if messages is None:
            raise Exception("At least one prompt is required")
        
        result = openai.ChatCompletion.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 16,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            n=self.count.value if self.count.value is not None else 1
        )
        

        if self.count.value is None or self.count.value == 1:
            response = result['choices'][0]['message']
            messages.append(response)
        self.completion.value = result['choices'][0]['message']['content']
        self.out_conversation.value = messages


@xai_component
class OpenAIEdit(Component):
    """Edits text using a specified model from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/edits/create)

    ##### inPorts:
    - model_name: Name of the model to be used for text editing.
    - prompt: The initial text to edit from.
    - instruction: Instructions for the edit.
    - count: Number of edited texts to generate.
    - temperature: Controls randomness of the output text.

    ##### outPorts:
    - edited: The edited text.
    """
    
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
    """Creates images from text using a specified model from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/images/create)

    ##### inPorts:
    - prompt: The text from which to generate images.
    - image_count: Number of images to generate. Default `1`.
    - size: The size of the generated images. Default "256x256".

    ##### outPorts:
    - image_urls: The URLs of the generated images.
    """

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
    """Downloads images from provided URLs.

    ##### inPorts:
    - image_urls: List of URLs of images to download.
    - file_path: List of file paths where the images will be saved.
    """

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
    """Creates variations of an image using a specified model from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/images/create-variation)

    ##### inPorts:
    - image_path: Path to the image file for which variations will be created.
    - image_count: Number of image variations to generate. Default `1`.
    - size: The size of the generated images. Default "256x256".

    ##### outPorts:
    - image_urls: The URLs of the generated image variations.
    """

    image_path: InCompArg[str]
    image_count: InArg[int]
    size: InArg[str]
    image_urls: OutArg[list]

    def execute(self, ctx) -> None:
        result = openai.Image.create_variation(
            image=open(self.image_path.value, "rb"),
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d['url'] for d in result['data']]
        
@xai_component
class OpenAIImageEdit(Component):

    """Edits an image using a specified model from OpenAI.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/images/create-edit)

    ##### inPorts:
    - prompt: The instruction for the image edit.
    - image: The original image to be edited.
    - mask: An optional mask for the edit.
    - image_count: Number of edited images to generate. Default `1`.
    - size: The size of the edited images. Default "256 x 256"

    ##### outPorts:
    - image_urls: The URLs of the edited images.
    """

    prompt:InCompArg[str]
    image: InCompArg[any]
    mask: InArg[any]
    image_count: InArg[int]
    size: InArg[str]
    image_urls: OutArg[list]

    def execute(self, ctx) -> None:
        result = openai.Image.create_edit(
            image=self.image.value,
            mask=self.mask.value,
            prompt=self.prompt.value,
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d['url'] for d in result['data']]
        
@xai_component
class TakeNthElement(Component):
    """Takes the nth element from a list.

    ##### inPorts:
    - values: List from which the nth element will be taken.
    - index: Index of the element to take.

    ##### outPorts:
    - out: The nth element of the list.
    """

    values: InCompArg[list]
    index: InCompArg[int]
    out: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.out.value = self.values.value[self.index.value]


@xai_component
class FormatConversation(Component):
    """Formats a conversation by appending messages to it.

    ##### inPorts:
    - prev_conversation: List of previous conversation messages.
    - system_prompt: Message to be appended from the system.
    - user_prompt: Message to be appended from the user.
    - faux_assistant_prompt: Message to be appended from the assistant.
    - input_prompt: Message to be appended from the user or system.
    - input_is_system: Boolean indicating whether the input prompt is from the system.
    - args: Arguments for formatting the messages.

    ##### outPorts:
    - out_messages: The formatted conversation.
    """
    prev_conversation: InArg[list]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    faux_assistant_prompt: InArg[str]
    input_prompt: InArg[str]
    input_is_system: InArg[bool]
    args: InArg[dict]
    out_messages: OutArg[list]
    
    def execute(self, ctx) -> None:
        conversation = [] if self.prev_conversation.value is None else self.prev_conversation.value
        format_args = {} if self.args.value is None else self.args.value
        
        
        if self.system_prompt.value is not None:
            conversation.append(self.make_msg('system', self.system_prompt.value.format(**format_args)))
            
        if self.user_prompt.value is not None:
            conversation.append(self.make_msg('user', self.user_prompt.value.format(**format_args)))
        
        if self.faux_assistant_prompt.value is not None:
            conversation.append(self.make_msg('assistant', self.faux_assistant_prompt.value.format(**format_args)))

        if self.input_prompt.value is not None:
            conversation.append(self.make_msg('system' if self.input_is_system.value else 'user', self.input_prompt.value.format(**format_args)))
        
        self.out_messages.value = conversation
        
        
    def make_msg(self, role, msg) -> dict:
        return { 'role': role, 'content': msg }

@xai_component
class AppendConversationResponse(Component):
    """Appends a response from the assistant to a conversation.

    ##### inPorts:
    - conversation: List of current conversation messages.
    - assistant_message: Message to be appended from the assistant.

    ##### outPorts:
    - out_conversation: The conversation including the assistant's response,
        ie: conversation + [{ 'role': 'assistant', 'content': assistant_message }]
    """
    conversation: InCompArg[list]
    assistant_message: InCompArg[str]
    out_conversation: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = self.conversation.value + [{ 'role': 'assistant', 'content': self.assistant_message.value}]
        self.out_conversation.value = ret

        
@xai_component
class JoinConversations(Component):
    """Appends multiple conversation lists into a single list.

    ##### inPorts:
    - conversation_1: First conversation to join.
    - conversation_2: Second conversation to join.
    - conversation_3: Third conversation to join.

    ##### outPorts:
    - out_conversation: The joined conversation.
    """

    conversation_1: InArg[list]
    conversation_2: InArg[list]
    conversation_3: InArg[list]
    
    out_conversation: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = []
        
        if self.conversation_1.value is not None:
            ret = ret + self.conversation_1.value
        if self.conversation_2.value is not None:
            ret = ret + self.conversation_2.value
        if self.conversation_3.value is not None:
            ret = ret + self.conversation_3.value
            
        self.out_conversation.value = ret


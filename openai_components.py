from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import openai
from openai import OpenAI

import os
import requests
import shutil
import base64

class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        for message in self.conversation_history:
            print(f"{message['role']}: {message['content']}\n\n")

def image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{base64_image}"
        return data_uri

@xai_component
class OpenAIMakeConversation(Component):
    """Creates a conversation object to hold conversation history.
    """
    prev: InArg[list]
    system_msg: InArg[str]
    user_msg: InArg[str]
    user_img: InArg[str]
    assistant_msg: InArg[str]
    function_msg: InArg[str]

    conversation: OutArg[list]

    def execute(self, ctx) -> None:
        conv = Conversation()

        if self.prev.value is not None:
            if isinstance(self.prev.value, list):
                conv.conversation_history.extend(self.prev.value)
            else:
                conv.conversation_history.extend(self.prev.value.conversation_history)
        if self.system_msg.value is not None:
            conv.add_message("system", self.system_msg.value)
        if self.user_msg.value is not None and self.user_img.value is None:
            conv.add_message("user", self.user_msg.value)
        if self.user_img.value is not None:
            image_url = image_to_data_uri(self.user_img.value)

            conv.add_message("user", [{ "type": "text", "text": self.user_msg.value }, { "type": "image_url", "image_url": { "url": image_url } } ])
        if self.assistant_msg.value is not None:
            conv.add_message("assistant", self.assistant_msg.value)
        if self.function_msg.value is not None:
            conv.add_message("function", self.function_msg.value)

        self.conversation.value = conv.conversation_history

@xai_component
class OpenAIAuthorize(Component):
    """Sets the organization and API key for the OpenAI client and creates an OpenAI client.

    This component checks if the API key should be fetched from the environment variables or from the provided input. 
    It then creates an OpenAI client using the API key and stores the client in the context (`ctx`) for use by other components.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/authentication)

    ##### inPorts:
    - organization: Organization name id for OpenAI API.
    - api_key: API key for the OpenAI API.
    - from_env: Boolean value indicating whether the API key is to be fetched from environment variables. 

    """
    organization: InArg[secret]
    base_url: InArg[str]
    api_key: InArg[secret]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        if self.from_env.value:
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = self.api_key.value

        client = OpenAI(
            api_key=api_key,
            organization=self.organization.value,
            base_url=self.base_url.value
        )
        ctx['client'] = client
        ctx['openai_api_key'] = api_key

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
        client = ctx['client']
        self.models.value = client.models.list()



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
        client = ctx['client']
        self.model.value = client.models.retrieve(self.model_name.value)

@xai_component
class OpenAIImageInference(Component):
    """
    Infers the content of an image using OpenAI's Vision capabilities.

    ##### inPorts:
    - model_name: The name of the OpenAI model to be used for inference.
    - image_input: Path to the image file (local path or URL). The component determines if it's a URL or a file path.
    - detail: Level of detail for inference. Options: "low", "high", or "auto". Default is "low".
    - input_prompt: Optional. A custom prompt to specify the question for the image inference. Default is "What is in this image?".

    ##### outPorts:
    - inference: The model's interpretation of the image.
    """

    model_name: InCompArg[str]
    image_input: InCompArg[str]
    detail: InArg[str]
    input_prompt: InArg[str]
    inference: OutArg[str]

    def execute(self, ctx) -> None:
        client = ctx.get("client")
        if client is None:
            raise ValueError("OpenAI client not found in context. Ensure OpenAI authorization is configured.")

        image_content = None
        input_value = self.image_input.value

        # Check if the input is a URL
        if input_value.startswith("http://") or input_value.startswith("https://"):
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": input_value,
                    "detail": self.detail.value if self.detail.value else "low"
                }
            }
        # Check if the input is a valid file path
        elif os.path.isfile(input_value):
            data_uri = image_to_data_uri(input_value)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": data_uri,
                    "detail": self.detail.value if self.detail.value else "low"
                }
            }
        else:
            raise ValueError("Image input must be a valid URL or file path. The provided input is invalid.")

        # Use the custom input prompt if provided, otherwise use the default
        prompt_text = self.input_prompt.value if self.input_prompt.value else "What is in this image?"

        # The image_content already has the structure: {"type": "image_url", "image_url": {"url": ..., "detail": ...}}
        # For the new API, the image part needs to be directly image_content's "image_url" value.
        # And the type for the image part should be "input_image".
        
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": image_content["image_url"]}
                ]
            }
        ]

        try:
            result = client.responses.create(
                model=self.model_name.value,
                input=input_payload
            )
            self.inference.value = result.output_text
        except Exception as e:
            raise RuntimeError(f"Failed to infer image: {e}")

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
        client = ctx['client']
        messages = [{"role": "user", "content": self.prompt.value}]
        result = client.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 16,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            n=self.count.value if self.count.value is not None else 1
        )

        if self.count.value is None or self.count.value == 1:
            self.completion.value = result.choices[0].message.content
        else:
            self.completion.value = [choice.message.content for choice in result.choices]

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
        if self.conversation.value is not None:
            messages = self.conversation.value
        else:
            messages = []
        
        if self.system_prompt.value is not None:            
            messages.append({"role": "system", "content": self.system_prompt.value})
        if self.user_prompt.value is not None:
            messages.append({"role": "user", "content": self.user_prompt.value })
        
        if not messages:
            raise Exception("At least one prompt is required")
        
        print("sending")
        for message in messages:
            print(message)
        
        client = ctx['client']
        result = client.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 128,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            n=self.count.value if self.count.value is not None else 1
        )
        

        if self.count.value is None or self.count.value == 1:
            response = result.choices[0].message
            messages.append({"role": "assistant", "content": response.content})
        self.completion.value = result.choices[0].message.content
        self.out_conversation.value = messages


@xai_component
class OpenAIStreamChat(Component):
    """Interacts with a specified model from OpenAI in a conversation, streams the response.

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
    completion_stream: OutArg[any]
    
    
    def execute(self, ctx) -> None:
        if self.conversation.value is not None:
            messages = self.conversation.value
        else:
            messages = []
        
        if self.system_prompt.value is not None:            
            messages.append({"role": "system", "content": self.system_prompt.value})
        if self.user_prompt.value is not None:
            messages.append({"role": "user", "content": self.user_prompt.value })
        
        if not messages:
            raise Exception("At least one prompt is required")
        
        print("sending")
        
        for message in messages:
            print(message)
        
        client = ctx['client']
        result = client.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 128,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            stream=True
        )
        
        def extract_text():
            for chunk in result:
                yield chunk.choices[0].delta.content

        self.completion_stream.value = extract_text()

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
        client = ctx['client']
        result = client.images.generate(
            prompt=self.prompt.value,
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d.url for d in result.data]



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
class DownloadImage(Component):
    """Downloads image from the provided URL.

    ##### inPorts:
    - image_url: URL of image to download.
    - file_path: file path where the image will be saved.
    """

    image_url: InCompArg[str]
    file_path: InCompArg[str]
    
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
        client = ctx['client']
        result = client.images.create_variation(
            image=open(self.image_path.value, "rb"),
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d.url for d in result.data]
        
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
        client = ctx['client']
        result = client.images.edit(
            image=self.image.value,
            mask=self.mask.value,
            prompt=self.prompt.value,
            n=self.image_count.value if self.image_count.value is not None else 1,
            size=self.size.value if self.size.value is not None else "256x256"
        )

        self.image_urls.value = [d.url for d in result.data]
        
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
    - system_message: Message to be appended from the assistant.
    - assistant_message: Message to be appended from the assistant.

    ##### outPorts:
    - out_conversation: The conversation including the assistant's response,
        ie: conversation + [{ 'role': 'assistant', 'content': assistant_message }]
    """
    conversation: InCompArg[list]
    system_message: InArg[str]
    assistant_message: InArg[str]
    out_conversation: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = self.conversation.value
        
        if self.system_message.value is not None:
            ret = ret + [{ 'role': 'assistant', 'content': self.assistant_message.value}]
        
        if self.assistant_message.value is not None:
            ret = ret + [{ 'role': 'assistant', 'content': self.assistant_message.value}]
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


import json
import re

def extract_json_object(string):
    # Regular expression to find a JSON object
    # It looks for a string that starts with { and ends with }, while ignoring any {} inside
    json_regex = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
    
    matches = re.finditer(json_regex, string)
    for match in matches:
        try:
            json_str = match.group(0)
            json_obj = json.loads(json_str)
            return json_obj  # Return the valid JSON object as a Python dictionary
        except json.JSONDecodeError:
            continue  # If it's not valid JSON, continue looking

    return None  # Return None if no valid JSON object is found

@xai_component
class OpenAIExtractJsonInString(Component):
    in_str: InCompArg[str]
    out_dict: OutArg[dict]
    
    def execute(self, ctx) -> None:
        self.out_dict.value = extract_json_object(self.in_str.value)

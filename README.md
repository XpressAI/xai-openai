<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>





<p align="center"><i>Xircuits Library for OpenAI: Seamlessly integrate AI to build and deploy intelligent solutions effortlessly.</i></p>


---

### Xircuits Component Library for OpenAI

This library connects OpenAI's models with Xircuits, making it easy to use features like text generation, image creation, and AI conversations.
## Table of Contents

- [Preview](#preview)
- [Prerequisites](#prerequisites)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview

### The Example:

![openai_sample](https://github.com/user-attachments/assets/1b557501-47e7-40fe-8152-65ea5e74694d)


### The Result

<img src="https://github.com/user-attachments/assets/d4b865b2-a37c-402e-8080-bb33f103453d" alt="openai_sample_result" />

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.

## Main Xircuits Components

### OpenAIAuthorize Component:

Sets the API key and organization for OpenAI, enabling access to its models.

<img src="https://github.com/user-attachments/assets/fed4ddb7-f751-4497-9c9c-d229c0aea1a4" alt="OpenAIAuthorize" width="200" height="150" />

### OpenAIGenerate Component:

Creates text completions using an OpenAI model, with adjustable parameters.

<img src="https://github.com/user-attachments/assets/889ddfbc-b62f-45d3-bade-ddb13505fc0d" alt="OpenAIGenerate" width="200" height="150" />

### OpenAIChat Component:

Conducts conversations with OpenAI models, maintaining conversation history.

<img src="https://github.com/user-attachments/assets/692e8620-8b52-42ab-8bb7-c3467c94b710" alt="OpenAIChat" width="200" height="200" />

### OpenAIGetModels Component:

Fetches a list of available OpenAI models for use in workflows.

### OpenAIGetModel Component:

Retrieves detailed information about a specific OpenAI model.

### OpenAIEdit Component:

Edits input text based on specific instructions provided.

### OpenAIImageCreate Component:

Generates images from text prompts using OpenAI's image models.

### OpenAIImageCreateVariation Component:

Creates alternate versions of an input image using OpenAI's variation tools.

### OpenAIImageEdit Component:

Modifies an input image based on a text prompt or a mask.

## Try the Examples

We have provided an example workflow to help you get started with the OpenAI component library. Give it a try and see how you can create custom OpenAI components for your applications.

### OpenAI Sample

This example authorizes OpenAI access using an API key and retrieves the list of available models. The models are then printed to the console.

## Installation
To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the OpenAI library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install openai
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory  
git clone https://github.com/XpressAI/xai-openai xai_components/xai_openai  
pip install -r xai_components/xai_openai/requirements.txt  
```

import image_handler
import random
import torch
import numpy as np
import argparse
import gradio as gr
import inspect
import os
import cv2
from PIL import Image
import uuid
import re
import requests
import json


from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics.


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human.
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

VISUAL_CHATGPT_PREFIX_CN = """Visual ChatGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 Visual ChatGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

Visual ChatGPT 能够处理和理解大量文本和图像。作为一种语言模型，Visual ChatGPT 不能直接读取图像，但它有一系列工具来完成不同的视觉任务。每张图片都会有一个文件名，格式为“image/xxx.png”，Visual ChatGPT可以调用不同的工具来间接理解图片。在谈论图片时，Visual ChatGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的图像文件时，Visual ChatGPT也知道图像可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实图像。 Visual ChatGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造图像内容和图像文件名。如果生成新图像，它将记得提供上次工具观察的文件名。

Human 可能会向 Visual ChatGPT 提供带有描述的新图形。描述帮助 Visual ChatGPT 理解这个图像，但 Visual ChatGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，Visual ChatGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

Visual ChatGPT 可以使用这些工具:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为Visual ChatGPT是一个文本语言模型，必须使用工具去观察图片而不是依靠想象。
推理想法和观察结果只对Visual ChatGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""


load_dotenv()
os.makedirs("image", exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def blend_gt2pt(old_image, new_image, sigma=0.15, steps=100):
    new_size = new_image.size
    old_size = old_image.size
    easy_img = np.array(new_image)
    gt_img_array = np.array(old_image)
    pos_w = (new_size[0] - old_size[0]) // 2
    pos_h = (new_size[1] - old_size[1]) // 2

    kernel_h = cv2.getGaussianKernel(old_size[1], old_size[1] * sigma)
    kernel_w = cv2.getGaussianKernel(old_size[0], old_size[0] * sigma)
    kernel = np.multiply(kernel_h, np.transpose(kernel_w))

    kernel[steps:-steps, steps:-steps] = 1
    kernel[:steps, :steps] = kernel[:steps, :steps] / kernel[steps - 1, steps - 1]
    kernel[:steps, -steps:] = kernel[:steps, -steps:] / kernel[steps - 1, -(steps)]
    kernel[-steps:, :steps] = kernel[-steps:, :steps] / kernel[-steps, steps - 1]
    kernel[-steps:, -steps:] = kernel[-steps:, -steps:] / kernel[-steps, -steps]
    kernel = np.expand_dims(kernel, 2)
    kernel = np.repeat(kernel, 3, 2)

    weight = np.linspace(0, 1, steps)
    top = np.expand_dims(weight, 1)
    top = np.repeat(top, old_size[0] - 2 * steps, 1)
    top = np.expand_dims(top, 2)
    top = np.repeat(top, 3, 2)

    weight = np.linspace(1, 0, steps)
    down = np.expand_dims(weight, 1)
    down = np.repeat(down, old_size[0] - 2 * steps, 1)
    down = np.expand_dims(down, 2)
    down = np.repeat(down, 3, 2)

    weight = np.linspace(0, 1, steps)
    left = np.expand_dims(weight, 0)
    left = np.repeat(left, old_size[1] - 2 * steps, 0)
    left = np.expand_dims(left, 2)
    left = np.repeat(left, 3, 2)

    weight = np.linspace(1, 0, steps)
    right = np.expand_dims(weight, 0)
    right = np.repeat(right, old_size[1] - 2 * steps, 0)
    right = np.expand_dims(right, 2)
    right = np.repeat(right, 3, 2)

    kernel[:steps, steps:-steps] = top
    kernel[-steps:, steps:-steps] = down
    kernel[steps:-steps, :steps] = left
    kernel[steps:-steps, -steps:] = right

    pt_gt_img = easy_img[pos_h : pos_h + old_size[1], pos_w : pos_w + old_size[0]]
    gaussian_gt_img = kernel * gt_img_array + (1 - kernel) * pt_gt_img  # gt img with blur img
    gaussian_gt_img = gaussian_gt_img.astype(np.int64)
    easy_img[pos_h : pos_h + old_size[1], pos_w : pos_w + old_size[0]] = gaussian_gt_img
    gaussian_img = Image.fromarray(easy_img)
    return gaussian_img


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split("\n")
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(" "))
        paragraphs = paragraphs[1:]
    return "\n" + "\n".join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split(".")[0].split("_")
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png"
    return os.path.join(head, new_file_name)


def getLlm():
    return OpenAI(temperature=0)
    # return "llm"


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        # self.device = device
        # self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        # self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.model = BlipForConditionalGeneration.from_pretrained(
        #     "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        # ).to(self.device)

    @prompts(
        name="Get Photo Description",
        description="useful when you want to know what is inside the photo. receives image_path as input. "
        "The input to this tool should be a string, representing the image_path. ",
    )
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        # self.device = device
        # self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = self.pipeFn
        # self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = "longbody, lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality"

    def pipeFn(self, prompt, negative_prompt=None):
        print("Text 2 image pipeline started")
        sd_endpoint = os.environ.get("SD_ENDPOINT", "https://8vltmrymi1y5st-7860.proxy.runpod.net/sdapi/v1/txt2img")
        sd_sampler = os.environ.get("SD_SAMPLER", "DPM++ 2M Karras")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        # Define the request data as a dictionary
        data = {
            "enable_hr": False,
            "denoising_strength": 0,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": "string",
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 50,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "restore_faces": False,
            "sampler_name": sd_sampler,
            "sampler_index": sd_sampler,
        }

        # Make the POST request
        response = requests.post(sd_endpoint, headers=headers, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code == 200:
            img = image_handler.turn_base64_to_png(image_handler.get_imagestr_from_sd_resp(response.content))
            print("Image retrieved")
            return [img]
        else:
            print("Error:", response.status_code, response.text)

    @prompts(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text):
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, self.n_prompt)[0]
        image.save(image_filename)
        print(f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class Text2Text:
    def __init__(self, device):
        print(f"Initializing Text2Text to {device}")
        # self.device = device
        # self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        # self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
        #                                                     torch_dtype=self.torch_dtype)
        # self.pipe.to(device)
        self.llm = getLlm()
        self.a_prompt = "best quality, extremely detailed"

    @prompts(
        name="Generate Text Response From User Input Text",
        description="Standard LLM",
    )
    def inference(self, text):
        prompt = text
        return self.llm(prompt)


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing Camel Bell, load_dict={load_dict}")
        # if "ImageCaptioning" not in load_dict:
        #     raise ValueError("You have to load ImageCaptioning as a basic function for Camel Bell")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k for k in inspect.signature(module.__init__).parameters.keys() if k != "self"
                }
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names}
                    )
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = getLlm()
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key="output")

    def init_agent(self, lang):
        self.memory.clear()  # clear previous history
        if lang == "English":
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VISUAL_CHATGPT_PREFIX,
                VISUAL_CHATGPT_FORMAT_INSTRUCTIONS,
                VISUAL_CHATGPT_SUFFIX,
            )
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VISUAL_CHATGPT_PREFIX_CN,
                VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN,
                VISUAL_CHATGPT_SUFFIX_CN,
            )
            place = "输入文字并回车，或者上传图片"
            label_clear = "清除"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={"prefix": PREFIX, "format_instructions": FORMAT_INSTRUCTIONS, "suffix": SUFFIX},
        )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(placeholder=place),
            gr.update(value=label_clear),
        )

    def run_text(self, text, state):
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer(), keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        print("result", res, "res output", res["output"])
        res["output"] = res["output"].replace("\\", "/")
        response = re.sub("(image/[-\w]*.png)", lambda m: f"![](/file={m.group(0)}) *{m.group(0)}*", res["output"])
        state = state + [(text, response)]
        print(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
            f"\nResponse: {response}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        state = state + [(f"![](/file={image_filename})*{image_filename}*", "")]
        print(
            f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, f"{txt} {image_filename} "


def run():
    # if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--load", type=str, default="ImageCaptioning_cuda:0,Text2Image_cuda:0")
    parser.add_argument("--load", type=str, default="Text2Text_cuda:0,Text2Image_cuda:0")
    args = parser.parse_args()
    # splits = args.load.split(",")
    # if len(splits) or splits[0] == "":
    #     raise ValueError("You have to load at least one model!")

    # image_handler.turn_base64_to_png(image_handler.get_imagestr_from_sd_resp_file("./response.json"), "./response.png")

    load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in args.load.split(",")}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:512px}") as demo:
        lang = gr.Radio(choices=["Chinese", "English"], value=None, label="Language")
        chatbot = gr.Chatbot(elem_id="chatbot", label="Camel Bell")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False
                )
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️", file_types=["image"])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0", server_port=7860)

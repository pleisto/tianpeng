from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from starlette.responses import RedirectResponse
from tianpeng.app.runner import ConversationBot
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.get("/")
def index():
    return RedirectResponse("/gradio", status_code=301)


app.mount("/gradio/image", StaticFiles(directory="image"), name="image")


bot = ConversationBot(load_dict={"Text2Text": "remote:0", "Text2Image": "remote:0"})
demo = gr.Blocks(css="#chatbot .overflow-y-auto{height:512px}", theme=gr.themes.Monochrome())
with demo:
    chatbot = gr.Chatbot(elem_id="chatbot", label="TianPeng")
    state = gr.State([])
    with gr.Row() as input_raws:
        with gr.Column(scale=0.61):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                container=False
            )
        with gr.Column(scale=0.13, min_width=0):
            clear = gr.Button("Clear")
        with gr.Column(scale=0.13, min_width=0):
            btn = gr.UploadButton(label="🖼️", file_types=["image"])
        with gr.Column(scale=0.13, min_width=0):
            file_btn = gr.UploadButton(label="file", file_types=["file"], elem_id="file")

    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
    file_btn.upload(bot.run_file, [file_btn, state, txt], [chatbot, state, txt])
    clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)


bot.init_agent()
app = gr.mount_gradio_app(app, demo, path="/gradio")


# pg_vector_util.init()
